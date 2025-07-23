import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from config import Config

config = Config()

class HierarchicalTemporalDataset(Dataset):
    """
    一个为分层模型准备数据的数据集类。
    它为每个样本生成三个部分：
    1. x_daily: 每日变化的快特征序列 (B, T, K_fast)
    2. x_yearly: 过去几年的慢特征上下文序列 (B, num_context_years, K_slow)
    3. y_target: 每日的目标值 (B, T)
    """
    def __init__(self, target_years, config, all_available_years_range):
        """
        初始化数据集。
        Args:
            target_years (range): 此数据集实例将为其生成样本的年份范围 (例如，训练年份或测试年份)。
            config: 包含模型配置的对象，我们主要使用 config.num_context_years。
            all_available_years_range (range): 数据集可以访问的所有年份的完整范围，用于查找历史上下文。
        """
        self.target_years = list(target_years)
        self.config = config
        self.all_available_years = list(all_available_years_range)
        
        # --- 特征维度定义 ---
        self.K_slow = 10 # 8个静态特征 + 2个年度特征
        self.K_fast = 5  # 5个快特征

        # --- 文件路径和归一化统计数据 ---
        self.input_files = {year: f"../../processed_data/TEM-MDM/temporal/input_{year}.npy" for year in self.all_available_years}
        self.output_files = {year: f"../../processed_data/TEM-MDM/temporal/output_{year}.npy" for year in self.target_years}
        self.mean_vals = np.load('../../data/TEM-MDM/mean_vals.npy')
        self.std_vals = np.load('../../data/TEM-MDM/std_vals.npy')


        # --- 关键改动 1: 预计算并缓存年度特征 ---
        # 为了避免在 __getitem__ 中反复读写文件，我们在初始化时一次性提取所有年份的慢特征。
        self.yearly_features_cache = self._precompute_yearly_features()

        # --- 关键改动 2: 仅基于目标年份计算长度 ---
        # 数据集的总长度由 target_years 中的样本总数决定。
        self.data_lengths = [np.load(self.input_files[year], mmap_mode='r').shape[0] for year in self.target_years]
        self.cumsum_lengths = np.cumsum([0] + self.data_lengths)

    def _precompute_yearly_features(self):
        """
        一个内部辅助函数，用于加载每个年份的输入数据，提取慢变特征。
        这只在初始化时运行一次，以提高效率。
        """
        cache = {}
        # 分别为慢特征和快特征切分归一化统计量
        mean_slow = self.mean_vals[:self.K_slow]
        std_slow = self.std_vals[:self.K_slow]

        for year in self.all_available_years:
            # 慢特征在一年中是恒定的，所以我们只取第一天的数据即可
            full_year_data = np.load(self.input_files[year], mmap_mode='r')
            slow_features_for_year = full_year_data[:, 0, :self.K_slow]
            slow_features_for_year = torch.tensor(slow_features_for_year, dtype=torch.float32)
            
            # 归一化并存入缓存
            normalized_slow_features = (slow_features_for_year - mean_slow) / std_slow
            cache[year] = torch.tensor(normalized_slow_features, dtype=torch.float32)
        return cache

    def __len__(self):
        return self.cumsum_lengths[-1]

    def __getitem__(self, idx):
        # 1. 定位样本：找到索引idx对应的年份和在该年内的文件索引
        year_list_idx = np.searchsorted(self.cumsum_lengths, idx, side="right") - 1
        file_idx = idx - self.cumsum_lengths[year_list_idx]
        actual_year = self.target_years[year_list_idx]

        # 2. 准备 X_daily 和 Y_target
        # 加载对应样本的完整输入数据
        full_input_sample = np.load(self.input_files[actual_year], mmap_mode='r')[file_idx]
        
        # 归一化整个样本
        normalized_input = (full_input_sample - self.mean_vals) / self.std_vals
        
        # 切分出快特征作为 X_daily
        x_daily = normalized_input[:, self.K_slow:]
        
        # 加载并归一化目标值
        y_target = np.load(self.output_files[actual_year], mmap_mode='r')[file_idx] / 23

        # 3. 准备 X_yearly：构建年度上下文序列
        context_vectors = []
        for i in range(1, self.config.num_context_years + 1):
            context_year = actual_year - i
            
            # 处理边界情况：如果需要的历史年份超出了数据范围，我们使用已知最早年份的数据作为填充
            if context_year in self.yearly_features_cache:
                vec = self.yearly_features_cache[context_year][file_idx]
            else:
                earliest_year = self.all_available_years[0]
                vec = self.yearly_features_cache[earliest_year][file_idx]
            context_vectors.append(vec)
        
        # 将向量列表堆叠成一个张量
        x_yearly = torch.stack(context_vectors)
        print("target", y_target.shape)
        print("x_daily", x_daily.shape)
        exit()

        return (
            torch.tensor(x_daily, dtype=torch.float32).to(config.device),
            torch.tensor(x_yearly, dtype=torch.float32).to(config.device),
            torch.tensor(y_target, dtype=torch.float32).to(config.device)
        )

# 年度编码器 (已在 3.2 节定义)
class AnnualEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(AnnualEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x_yearly):
        _, h_n = self.gru(x_yearly)
        return h_n.squeeze(0)

# 交叉注意力模块
class CrossAttention(nn.Module):
    """
    一个简单的交叉注意力模块，用于计算日度解码器状态和年度上下文之间的对齐。
    """
    def __init__(self, daily_hidden_dim, annual_context_dim):
        super(CrossAttention, self).__init__()
        # 线性层，将日度隐藏状态（查询）映射到与年度上下文（键）相同的维度
        self.query_layer = nn.Linear(daily_hidden_dim, annual_context_dim)
        
    def forward(self, daily_hidden, annual_context):
        # daily_hidden (query): (batch_size, daily_hidden_dim)
        # annual_context (key/value): (batch_size, annual_context_dim)
        
        # 1. 投射查询
        query = self.query_layer(daily_hidden) # -> (batch_size, annual_context_dim)
        
        # 2. 计算注意力分数 (scaled dot-product)
        # 为了进行批处理矩阵乘法，我们需要增加一个维度
        query = query.unsqueeze(1) # -> (batch_size, 1, annual_context_dim)
        # 将 annual_context 视为只有一个键/值对的序列
        annual_context_unsqueezed = annual_context.unsqueeze(2) # -> (batch_size, annual_context_dim, 1)
        
        # energy 的形状: (batch_size, 1, 1)
        energy = torch.bmm(query, annual_context_unsqueezed)
        
        # 在这个简单的场景下，只有一个注意力权重，恒为1，softmax是可选的
        # 但为了通用性，我们保留它
        attention_weights = F.softmax(energy, dim=-1)
        
        # 3. 计算加权上下文
        # attended_context 的形状: (batch_size, 1, annual_context_dim)
        attended_context = torch.bmm(attention_weights, annual_context.unsqueeze(1))
        
        return attended_context.squeeze(1), attention_weights.squeeze()

# 日度解码器
class DailyDecoder(nn.Module):
    def __init__(self, daily_input_dim, daily_hidden_dim, annual_context_dim, output_dim=1):
        super(DailyDecoder, self).__init__()
        self.lstm = nn.LSTM(daily_input_dim, daily_hidden_dim, batch_first=True)
        self.attention = CrossAttention(daily_hidden_dim, annual_context_dim)
        
        # 最终的预测头，输入是拼接后的LSTM输出和注意力上下文
        self.fc = nn.Linear(daily_hidden_dim + annual_context_dim, output_dim)

    def forward(self, x_daily, annual_context):
        # x_daily: (batch_size, L, K_fast)
        # annual_context: (batch_size, annual_context_dim)
        
        # LSTM处理日度序列
        lstm_out, (h_n, c_n) = self.lstm(x_daily)
        
        # 我们使用最后一个时间步的LSTM隐藏状态作为查询
        last_hidden_state = h_n.squeeze(0) # -> (batch_size, daily_hidden_dim)
        
        # 通过注意力机制计算加权的年度上下文
        attended_context, attention_weights = self.attention(last_hidden_state, annual_context)
        
        # 拼接LSTM输出和注意力上下文
        combined = torch.cat((last_hidden_state, attended_context), dim=1)
        
        # 通过全连接层得到最终预测
        prediction = self.fc(combined)
        
        return prediction, attention_weights

# 完整的分层模型
class Hierarchical(nn.Module):
    def __init__(self, annual_input_dim, annual_hidden_dim, daily_input_dim, daily_hidden_dim, output_dim=1):
        super(Hierarchical, self).__init__()
        self.encoder = AnnualEncoder(annual_input_dim, annual_hidden_dim)
        self.decoder = DailyDecoder(daily_input_dim, daily_hidden_dim, annual_hidden_dim, output_dim)

    def forward(self, x_daily, x_yearly):
        # 1. 从年度数据编码长期上下文
        annual_context = self.encoder(x_yearly)
        
        # 2. 使用长期上下文和日度数据进行解码和预测
        prediction, attention_weights = self.decoder(x_daily, annual_context)
        print("predication shape", prediction.shape)
        
        return prediction, prediction