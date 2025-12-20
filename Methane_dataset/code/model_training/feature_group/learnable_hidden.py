import torch
import torch.nn as nn

class VanillaLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(VanillaLSTMCell, self).__init__()
        self.hidden_size = hidden_size

        # 为了效率，我们将四个线性变换合并为一个
        # W_f, W_i, W_c, W_o 的权重可以一次性计算
        self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, input_tensor, prev_state):
        # prev_state 是一个元组 (h_{t-1}, c_{t-1})
        h_prev, c_prev = prev_state

        # 将当前输入和上一个隐藏状态拼接
        combined = torch.cat([input_tensor, h_prev], dim=1)
        
        # 一次性计算四个门和候选细胞状态的线性部分
        gates = self.linear(combined)
        
        # 将结果切分成四份
        # f_t, i_t, c_tilde, o_t
        f, i, c_tilde, o = torch.chunk(gates, 4, dim=1)

        # 应用激活函数
        f_t = torch.sigmoid(f)
        i_t = torch.sigmoid(i)
        c_tilde_t = torch.tanh(c_tilde)
        o_t = torch.sigmoid(o)

        # 计算新的细胞状态 c_t
        # c_t = (遗忘门 * 上一个细胞状态) + (输入门 * 新的候选状态)
        c_t = f_t * c_prev + i_t * c_tilde_t

        # 计算新的隐藏状态 h_t
        # h_t = 输出门 * tanh(新的细胞状态)
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

class VanillaLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(VanillaLSTMLayer, self).__init__()
        self.hidden_size = hidden_size
        self.cell = VanillaLSTMCell(input_size, hidden_size)

    def forward(self, input_sequence, initial_state=None):
        # input_sequence: (batch_size, seq_length, input_size)
        batch_size, seq_length, _ = input_sequence.shape

        if initial_state is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=input_sequence.device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=input_sequence.device)
        else:
            h_t, c_t = initial_state
            
        outputs = []
        # 逐个时间步进行处理
        for t in range(seq_length):
            input_t = input_sequence[:, t, :]
            h_t, c_t = self.cell(input_t, (h_t, c_t))
            outputs.append(h_t.unsqueeze(1))
        
        # 将所有时间步的输出拼接起来
        outputs = torch.cat(outputs, dim=1) # (batch_size, seq_length, hidden_size)
        
        return outputs, (h_t, c_t)