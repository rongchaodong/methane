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
        self.input_files = {year: f"../../processed_data/TEM-MDM/sequence/temporal/input_{year}.npy" for year in self.all_available_years}
        self.output_files = {year: f"../../processed_data/TEM-MDM/sequence/temporal/output_{year}.npy" for year in self.target_years}
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
            cache[year] = normalized_slow_features.detach()
        return cache

    def __len__(self):
        return self.cumsum_lengths[-1]

    def __getitem__(self, idx):
        # 1. 定位样本：找到索引idx对应的年份和在该年内的文件索引
        year_list_idx = np.searchsorted(self.cumsum_lengths, idx, side="right") - 1
        file_idx = idx - self.cumsum_lengths[year_list_idx]
        actual_year = self.target_years[year_list_idx]
        # print("year_list_idx", year_list_idx, " year:", actual_year, " data length: ", self.data_lengths[year_list_idx])
        # print("file idx:", file_idx)

        # 2. 准备 X_daily 和 Y_target
        # 加载对应样本的完整输入数据
        full_input_sample = np.load(self.input_files[actual_year], mmap_mode='r')[file_idx]
        # print("this year data:", full_input_sample[0])
        
        # 归一化整个样本
        normalized_input = (full_input_sample - self.mean_vals) / self.std_vals
        # print("this year data:", normalized_input[0][:10])
        
        # 切分出快特征作为 X_daily
        x_daily = normalized_input[:, self.K_slow:]
        
        # 加载并归一化目标值
        y_target = np.load(self.output_files[actual_year], mmap_mode='r')[file_idx] / 23

        # 3. 准备 X_yearly：构建年度上下文序列
        context_vectors = []
        for i in range(1, self.config.num_context_years + 1):
            context_year = actual_year - i
            # print(self.mean_vals.shape)
            # print(self.std_vals.shape)
            # print("context year", context_year, " context year length", self.data_lengths[year_list_idx - i])
            # print("context year true data:", np.load(self.input_files[context_year], mmap_mode='r')[file_idx][0][:10])
            
            # 处理边界情况：如果需要的历史年份超出了数据范围，我们使用已知最早年份的数据作为填充
            if context_year in self.yearly_features_cache:
                vec = self.yearly_features_cache[context_year][file_idx]
            else:
                earliest_year = self.all_available_years[0]
                vec = self.yearly_features_cache[earliest_year][file_idx]
            # print("context vec of :", context_year, " ", vec)
            context_vectors.append(vec)
        
        # 将向量列表堆叠成一个张量
        x_yearly = torch.stack(context_vectors)
        # print("target", y_target.shape)
        # print("x_daily", x_daily.shape)
        # print("year_daily", x_yearly.shape)
        # exit()

        return (
            torch.tensor(x_daily, dtype=torch.float32).to(config.device),
            x_yearly.to(config.device),
            torch.tensor(y_target, dtype=torch.float32).to(config.device)
        )

# 年度编码器 (已在 3.2 节定义)
class AnnualEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(AnnualEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x_yearly):
        _, h_n = self.gru(x_yearly)
        # print("GRU shape:", h_n.shape, h_n.squeeze(0).shape)
        return h_n.squeeze(0)

class CrossAttention(nn.Module):
    def __init__(self, daily_hidden_dim, annual_context_dim):
        super(CrossAttention, self).__init__()
        self.query_layer = nn.Linear(daily_hidden_dim, annual_context_dim)

    def forward(self, daily_hidden, annual_context):
        # daily_hidden: [B, 365, H_d]
        # annual_context: [B, H_y]
        B, T, H_d = daily_hidden.shape
        annual_context = annual_context.unsqueeze(1)  # [B, 1, H_y]
        query = self.query_layer(daily_hidden)        # [B, 365, H_y]

        # Dot-product attention score: [B, 365, 1]
        energy = torch.bmm(query, annual_context.transpose(1, 2)) / (H_d ** 0.5)
        attention_weights = F.softmax(energy, dim=1)  # [B, 365, 1]

        # context = weighted sum of annual_context (broadcasted): [B, 365, H_y]
        context = attention_weights * annual_context

        return context, attention_weights.squeeze(-1)  # [B, 365, H_y], [B, 365]


class DailyDecoder(nn.Module):
    def __init__(self, daily_input_dim, daily_hidden_dim, annual_context_dim, output_dim=1, num_heads=2, dropout=0.1):
        super(DailyDecoder, self).__init__()
        self.lstm = nn.LSTM(daily_input_dim, daily_hidden_dim, num_layers=3, batch_first=True)
        
        # 将年度上下文扩展为序列的注意力机制,可以扩展到更大的维度降低信息损失
        self.attn_proj = nn.Linear(annual_context_dim, daily_hidden_dim)  # 投射到与 LSTM 输出一致的维度
        self.multihead_attn = nn.MultiheadAttention(daily_hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(daily_hidden_dim)

        self.fc = nn.Linear(daily_hidden_dim, output_dim)

    def forward(self, x_daily, annual_context):
        # x_daily: [B, 365, K_fast]
        # annual_context: [B, annual_context_dim]

        lstm_out, _ = self.lstm(x_daily)  # [B, 365, H]

        # annual_context: [B, H_y] -> [B, 1, H_y] -> [B, 1, H]
        annual_context_proj = self.attn_proj(annual_context).unsqueeze(1)  # [B, 1, H]

        # 使用年度上下文作为 key 和 value，LSTM 输出作为 query
        # 注意 MultiheadAttention 的输入要是 [B, T, H]
        attn_output, attn_weights = self.multihead_attn(
            query=lstm_out,
            key=annual_context_proj.expand(-1, lstm_out.size(1), -1),  # [B, 365, H]
            value=annual_context_proj.expand(-1, lstm_out.size(1), -1) # [B, 365, H]
        )  # 输出: [B, 365, H], attn_weights: [B, 365, 365]

        # 残差 + LayerNorm（可选）
        combined = self.norm(lstm_out + attn_output)  # [B, 365, H]

        prediction = self.fc(combined).squeeze(-1)  # [B, 365]

        return prediction, attn_weights.mean(dim=1)  # 平均注意力权重（[B, 365]）


# class DailyDecoder(nn.Module):
#     def __init__(self, daily_input_dim, daily_hidden_dim, annual_context_dim, output_dim=1):
#         super(DailyDecoder, self).__init__()
#         self.lstm = nn.LSTM(daily_input_dim, daily_hidden_dim, batch_first=True)
#         self.attention = CrossAttention(daily_hidden_dim, annual_context_dim)
#         self.fc = nn.Linear(daily_hidden_dim + annual_context_dim, output_dim)

#     def forward(self, x_daily, annual_context):
#         lstm_out, _ = self.lstm(x_daily)  # [B, 365, H_d]

#         # 每天都与 annual_context 做 attention
#         attended_context, attention_weights = self.attention(lstm_out, annual_context)  # [B, 365, H_y]

#         # 拼接
#         combined = torch.cat((lstm_out, attended_context), dim=2)  # [B, 365, H_d + H_y]

#         prediction = self.fc(combined).squeeze(-1)  # [B, 365]

#         return prediction, attention_weights


# 完整的分层模型
class Hierarchical(nn.Module):
    def __init__(self, annual_input_dim, annual_hidden_dim, daily_input_dim, daily_hidden_dim, output_dim=1):
        super(Hierarchical, self).__init__()
        self.encoder = AnnualEncoder(annual_input_dim, annual_hidden_dim)
        self.decoder = DailyDecoder(daily_input_dim, daily_hidden_dim, annual_hidden_dim, output_dim)

    def forward(self, x_daily, x_yearly):
        # print("input shape: daily:", x_daily.shape, "yearly:", x_yearly.shape)
        # 1. 从年度数据编码长期上下文
        annual_context = self.encoder(x_yearly)
        # print("annaual context shape:", annual_context.shape)
        
        # 2. 使用长期上下文和日度数据进行解码和预测
        prediction, attention_weights = self.decoder(x_daily, annual_context)
        
        return prediction, prediction