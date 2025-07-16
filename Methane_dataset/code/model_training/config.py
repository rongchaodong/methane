import torch

class Config:
    def __init__(self, batch_size=128, enc_in=15, n_hidden=8, ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.enc_in = enc_in
        self.hidden_size = n_hidden
        self.batch_first = True 
        self.num_layers = 3
        self.input_size = 15
        self.output_size = 1
        self.seq_len = 365
        self.initial_forget_bias = 0
        self.input_size_dyn = 5
        self.input_size_stat = 10
        self.dropout = 0.2
        self.concat_static = False
        self.no_static = False
        # TCN
        self.num_channels = [1,1,1]
        self.kernel_size = 5
        # Transformer
        self.d_model = 64
        self.e_layers = 3
        self.d_layers = 1
        self.n_heads = 4
        self.factor = 3
        self.embed = "time"
        self.freq = 'd'
        self.d_model = 64
        self.d_ff = 4*self.d_model
        self.c_out = 1
        self.activation = 'relu'
        self.output_attention = False
        self.pred_len = 365
        self.task_name = 'long_term_forecast'
        # --- NEW: Hierarchical Model Config ---
        self.max_num_years = 40  # 训练+测试的总年份跨度，用于位置编码
        self.meta_n_heads = 2      # 元转换器的头数
        self.meta_num_layers = 2   # 元转换器的层数