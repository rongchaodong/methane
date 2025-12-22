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
        self.d_ff = 4*self.d_model
        self.c_out = 1
        self.activation = 'relu'
        self.output_attention = False
        self.pred_len = 365
        self.task_name = 'long_term_forecast'

        # TimeMixer
        self.down_sampling_window = 2   # default is 2
        self.channel_independence = 0   # 0: False 15->1，1: True 15->15, indenpendent processing per channel
        self.moving_avg = 25            # daily data is 30 to divide the 365 into nearly 12 months
        
        self.down_sampling_layers = 3   # (365 -> 182 -> 91 -> 45)
        self.down_sampling_method = 'avg' # 'avg', 'max', 'conv', defalut 'avg'
        self.decomp_method = 'moving_avg' # 'moving_avg' or 'dft_decomp'
        self.use_norm = 1        # 1: use Normalize, 0: otherwise

        # P-sLSTM
        self.channel = enc_in
        self.conv1d_kernel_size = 4 # default 4 in the code
        self.patch_size = 16
        self.stride = 8         # default 8 in the code
        self.embedding_dim = self.d_model
        self.num_blocks = 2
        self.num_heads = 4

        # --- NEW: Hierarchical Model Config ---
        self.max_num_years = 40  
        self.meta_n_heads = 1      
        self.meta_num_layers = 1  
        self.att_inter_years = 2

        self.num_context_years = 4