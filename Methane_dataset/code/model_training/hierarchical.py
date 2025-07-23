import torch
import torch.nn as nn
import torch.nn.functional as F

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
class HierarchicalMethaneModel(nn.Module):
    def __init__(self, annual_input_dim, annual_hidden_dim, daily_input_dim, daily_hidden_dim, output_dim=1):
        super(HierarchicalMethaneModel, self).__init__()
        self.encoder = AnnualEncoder(annual_input_dim, annual_hidden_dim)
        self.decoder = DailyDecoder(daily_input_dim, daily_hidden_dim, annual_hidden_dim, output_dim)

    def forward(self, x_daily, x_yearly):
        # 1. 从年度数据编码长期上下文
        annual_context = self.encoder(x_yearly)
        
        # 2. 使用长期上下文和日度数据进行解码和预测
        prediction, attention_weights = self.decoder(x_daily, annual_context)
        
        return prediction