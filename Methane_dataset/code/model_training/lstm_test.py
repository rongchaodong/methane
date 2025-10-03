import torch
import torch.nn as nn
from torch.autograd import Variable
from learnable_hidden import VanillaLSTMLayer

# 定义输入数据
batch_size = 4
seq_length = 1
input_size = 8
hidden_size = 16

# 随机生成输入数据
input_data = torch.randn(batch_size, seq_length, input_size)

# 初始化自定义 LSTM 和 nn.LSTM
custom_lstm = VanillaLSTMLayer(input_size, hidden_size)
pytorch_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

# 初始化权重和偏置，使两者一致
with torch.no_grad():
    # 从 custom_lstm 中拆分权重
    custom_weight = custom_lstm.cell.linear.weight
    custom_bias = custom_lstm.cell.linear.bias

    # 检查 custom_bias 的形状是否正确
    assert custom_bias.shape[0] == 4 * hidden_size, "custom_bias 的长度必须为 4 * hidden_size"

    # 拆分输入到隐藏层的权重 (weight_ih_l0) 和隐藏层到隐藏层的权重 (weight_hh_l0)
    pytorch_lstm.weight_ih_l0.copy_(custom_weight[:, :input_size])  # 前 input_size 列
    pytorch_lstm.weight_hh_l0.copy_(custom_weight[:, input_size:])  # 后 hidden_size 列

    # 偏置直接复制
    pytorch_lstm.bias_ih_l0.copy_(custom_bias)  # PyTorch 的 bias_ih_l0 和 bias_hh_l0 默认相同
    pytorch_lstm.bias_hh_l0.copy_(custom_bias)

# 前向传播
custom_output, (custom_h_n, custom_c_n) = custom_lstm(input_data)
pytorch_output, (pytorch_h_n, pytorch_c_n) = pytorch_lstm(input_data)

# 比较输出
print("Custom LSTM Output:", custom_output)
print("PyTorch LSTM Output:", pytorch_output)
print("Output Difference:", torch.abs(custom_output - pytorch_output).max().item())