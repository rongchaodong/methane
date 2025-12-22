import sys
sys.path.append(f'Time-Series-Library/')
sys.path.append(f'Time-Series-Library/layers/')
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import Tuple
from Transformer_EncDec import Encoder, EncoderLayer
from SelfAttention_Family import FullAttention, AttentionLayer
from Embed import DataEmbedding_inverted, PatchEmbedding
from layers.Pyraformer_EncDec import Encoder as Pyra_Encoder

class LSTM(nn.Module):
    def __init__(self, configs):
        super(LSTM, self).__init__()
        self.hidden_size = configs.hidden_size
        self.num_layers = configs.num_layers
        self.lstm = nn.LSTM(configs.input_size, configs.hidden_size, num_layers=3, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(configs.hidden_size, configs.hidden_size * 4),
            nn.GELU(),
            nn.Linear(configs.hidden_size * 4, configs.output_size)
        )
        self.fc = nn.Linear(configs.hidden_size, configs.output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)  
        final_out = self.head(out)
        # final_out = self.fc(out)  
        return out, final_out.squeeze()
    
class EALSTM(nn.Module):
    def __init__(self, configs):
        super(EALSTM, self).__init__()
        self.configs = configs
        self.device = configs.device
        self.batch_size = configs.batch_size
        self.input_size_dyn = configs.enc_in - configs.input_size_stat
        self.input_size_stat = configs.input_size_stat
        self.hidden_size = configs.hidden_size
        self.batch_first = configs.batch_first
        self.initial_forget_bias = configs.initial_forget_bias

        self.weight_ih = nn.Parameter(torch.FloatTensor(self.input_size_dyn, 3 * self.hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(self.hidden_size, 3 * self.hidden_size))
        self.weight_sh = nn.Parameter(torch.FloatTensor(self.input_size_stat, self.hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(3 * self.hidden_size))
        self.bias_s = nn.Parameter(torch.FloatTensor(self.hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight_ih.data)
        nn.init.orthogonal_(self.weight_sh)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh.data = weight_hh_data

        nn.init.constant_(self.bias.data, val=0)
        nn.init.constant_(self.bias_s.data, val=0)

        if self.initial_forget_bias != 0:
            self.bias.data[:self.hidden_size] = self.initial_forget_bias

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.batch_first:
            x_d = x_d.transpose(0, 1)

        seq_len, batch_size, _ = x_d.size()

        h_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        c_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        h_x = (h_0, c_0)

        h_n, c_n = [], []
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))
        bias_s_batch = (self.bias_s.unsqueeze(0).expand(batch_size, *self.bias_s.size()))
        i = torch.sigmoid(torch.addmm(bias_s_batch, x_s, self.weight_sh))

        for t in range(seq_len):
            h_0, c_0 = h_x
            gates = (torch.addmm(bias_batch, h_0, self.weight_hh) +
                     torch.mm(x_d[t], self.weight_ih))
            f, o, g = gates.chunk(3, 1)
            c_1 = torch.sigmoid(f) * c_0 + i * torch.tanh(g)
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)
            h_n.append(h_1)
            c_n.append(c_1)

            h_x = (h_1, c_1)

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        if self.batch_first:
            h_n = h_n.transpose(0, 1)
            c_n = c_n.transpose(0, 1)

        return h_n, c_n

class MyEALSTM(nn.Module):
    def __init__(self,configs):
        super(MyEALSTM, self).__init__()

        self.configs = configs
        self.device = configs.device

        self.batch_size = configs.batch_size
        self.input_size_dyn = configs.enc_in - configs.input_size_stat
        self.input_size_stat = configs.input_size_stat
        self.hidden_size = configs.hidden_size
        self.batch_first = configs.batch_first
        self.initial_forget_bias = configs.initial_forget_bias
        self.dropout_rate = configs.dropout
        self.concat_static = configs.concat_static
        self.no_static = configs.no_static
        self.lstm = EALSTM(configs=configs)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(self.hidden_size, 1)


    def forward(self, x: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_d = x[:, :, self.input_size_stat:]
        x_s = x[:, 0, :self.input_size_stat] 
        if self.concat_static or self.no_static:
            h_n, c_n = self.lstm(x_d)
        else:
            h_n, c_n = self.lstm(x_d, x_s)
    
        h_n = self.dropout(h_n)
        out = self.fc(h_n)
        return h_n, out.squeeze(2)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # print('before tcn', x.shape)
        out = self.net(x.unsqueeze(2)).squeeze(2)
        # print('after tcn', out.shape)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, config):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(config.num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            layers += [TemporalBlock(config.seq_len, config.seq_len, config.kernel_size, stride=1, dilation=dilation_size,
                                     padding=(config.kernel_size-1) * dilation_size, dropout=config.dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class MultiTCN(nn.Module):
    def __init__(self, config):
        super(MultiTCN, self).__init__()
        self.tcn_blocks = nn.ModuleList([
            TemporalConvNet(config) for _ in range(config.input_size)
        ])

        self.mlp = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size), 
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.output_size) 
        )

    def forward(self, x):
        batch_size, seq_len, num_features = x.shape

        tcn_outputs = []
        for i in range(num_features):
            feature_x = x[:, :, i] 
            # print(feature_x.shape)
            tcn_output = self.tcn_blocks[i](feature_x)
            tcn_outputs.append(tcn_output)

        combined_tcn_output = torch.stack(tcn_outputs, dim=-1)

        final_output = self.mlp(combined_tcn_output).squeeze(2)
        return combined_tcn_output, final_output

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(config.input_size, config.hidden_size)  
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=4, batch_first=True), 
            num_layers=config.num_layers
        )
        self.fc_out = nn.Linear(config.hidden_size, config.output_size) 

    def make_mask(self, src):
        seq_len = src.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.bool()
        return mask
    
    def forward(self, src):
        # print(src.shape)
        mask = self.make_mask(src).to(src.device)
        src_emb = self.embedding(src) 
        transformer_out = self.transformer(src_emb, mask=mask)
        output = self.fc_out(transformer_out) 
        # print(output.shape) 
        
        return transformer_out, output.squeeze(2)
    
class iTransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(iTransformer, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(
                configs.d_model, configs.pred_len, bias=True)
        # Linear layer
        self.linear = nn.Linear(configs.enc_in, configs.c_out)
        self.relu = nn.ReLU()

    def forward(self,  x):
        # Normalization from Non-stationary Transformer
        _, _, N = x.shape

        # Embedding
        dec_out = self.enc_embedding(x, x_mark=None)
        dec_out, attns = self.encoder(dec_out, attn_mask=None)

        dec_emb = self.projection(dec_out).permute(0, 2, 1)[:, :, :N]
        dec_out = self.linear(dec_emb)
        # dec_out = self.relu(dec_out[:,:,0])
        dec_out = self.relu(dec_out)
        return dec_emb, dec_out[:,:,0]
    
class Pyraformer(nn.Module):
    """
    Pyraformer: Pyramidal attention to reduce complexity
    Paper link: https://openreview.net/pdf?id=0EXmFzUn5I
    """

    def __init__(self, configs, window_size=[4,4], inner_size=5):
        """
        window_size: list, the downsample window size in pyramidal attention.
        inner_size: int, the size of neighbour attention
        """
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model

        self.encoder = Pyra_Encoder(configs, window_size, inner_size)

        if self.task_name == 'long_term_forecast':
            self.projection = nn.Linear(
                (len(window_size)+1)*self.d_model, self.pred_len * configs.enc_in)
        self.linear = nn.Linear(configs.enc_in, configs.c_out)


    def forward(self, x_enc, x_mark_enc=None):
        enc_out = self.encoder(x_enc, x_mark_enc)[:, -1, :]
        dec_emb = self.projection(enc_out).view(
            enc_out.size(0), self.pred_len, -1)
        dec_out = self.linear(dec_emb[:, -self.pred_len:, :] )
        return dec_emb[:, -self.pred_len:, :], dec_out[:,:,0] # [B, L, D]
    

### TSMixer implmentation

class ResBlock(nn.Module):
    def __init__(self, configs):
        super(ResBlock, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.seq_len),
            nn.Dropout(configs.dropout)
        )

        self.channel = nn.Sequential(
            nn.Linear(configs.enc_in, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.enc_in),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)

        return x


class TSMixer(nn.Module):
    def __init__(self, configs):
        super(TSMixer, self).__init__()
        self.task_name = configs.task_name
        self.layer = configs.e_layers
        self.model = nn.ModuleList([ResBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.pred_len = configs.pred_len
        self.projection = nn.Linear(configs.seq_len, configs.pred_len)

        ## add a target projection layer for methane forecasting
        self.output_layer = nn.Linear(configs.enc_in, configs.c_out)

    def forecast(self, x_enc):

        # x: [B, L, D]
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        enc_out = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)
        enc_out = self.output_layer(enc_out)

        return enc_out

    def forward(self, x_enc):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :], dec_out.squeeze()  # [B, L, D], [B, L]
        else:
            raise ValueError('Only forecast tasks implemented yet')
        


# implementation of TimeMixer model
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k: int = 5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, k=self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('decompsition is error')

        if not configs.channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class TimeMixer(nn.Module):

    def __init__(self, configs):
        super(TimeMixer, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in

        if self.channel_independence:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            if self.channel_independence:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)

                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )
        self.final_layer = nn.Linear(configs.enc_in, configs.c_out)


    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc
    
    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def forecast(self, x_enc, x_mark_enc=None):

        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_list.append(x)
                    x_mark = x_mark.repeat(N, 1, 1)
                    x_mark_list.append(x_mark)
                else:
                    x_list.append(x)
                    x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def forward(self, x_enc, x_mark_enc = None, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc)
            dec_out = self.final_layer(dec_out).squeeze()
            # print('TimeMixer output shape:', dec_out.shape)
            return dec_out, dec_out
        else:
            raise ValueError('Other tasks implemented yet')
        

### implementation of PatchTST model

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class PatchTST(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        else:
            raise ValueError('Only forecast task are supported yet!')
        
        self.final_layer = nn.Linear(configs.enc_in, configs.c_out)

    def forecast(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            dec_out = self.final_layer(dec_out).squeeze()
            return dec_out[:, -self.pred_len:], dec_out  # [B, L, D]
        return None


### implementation of P-sLSTM model
from einops import rearrange

from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm.blocks.slstm.block import sLSTMBlockConfig
from xlstm.blocks.slstm.layer import sLSTMLayerConfig
from xlstm.components.feedforward import FeedForwardConfig

class P_sLSTM(nn.Module):

    def __init__(self, configs):
        super(P_sLSTM, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channel = configs.channel
        self.embedding_dim = configs.embedding_dim
        self.patch_size = configs.patch_size  # patch size
        self.stride = configs.stride  # the stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1  # the number of patches

        # xLSTM configuration
        cfg = xLSTMBlockStackConfig(
            # mlstm_block=mLSTMBlockConfig(
            #     mlstm=mLSTMLayerConfig(conv1d_kernel_size=configs.conv1d_kernel_size, 
            #                            qkv_proj_blocksize=4, num_heads=configs.num_heads)
            # ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(num_heads=configs.num_heads, 
                                       conv1d_kernel_size=configs.conv1d_kernel_size,
                                       bias_init="powerlaw_blockdependent",
                                       backend="vanilla"),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=self.seq_len,
            num_blocks=configs.num_blocks,
            embedding_dim=configs.embedding_dim,
            slstm_at="all"
        )

        self.embedding = nn.Linear(self.patch_size, self.embedding_dim)  
        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.projection = nn.Linear(self.embedding_dim * self.patch_num, self.pred_len)  
        self.final_layer = nn.Linear(configs.channel, configs.c_out)

    def forward(self, x):

        # x: [Batch, Input length, Channel]
        B, L, C = x.shape

        # Rearrange dimensions 
        x = rearrange(x, 'b l c -> b c l')

        # Create patches
        x = rearrange(x, 'b c l -> (b c) l')  # [(B * C), Num_patches, Patch_size]
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # [(B * C), Num_patches, Patch_size]

        x = self.embedding(x)  # [(B * C), Num_patches, Embedding_dim]

        x = self.xlstm_stack(x)  # [(B * C), Num_patches, Embedding_dim]

        x = x.flatten(1)  # [(B * C), Num_patches * Embedding_dim]

        x = self.projection(x)  # [(B * C), Pred_len]

        # Reshape back to b c l
        x = rearrange(x, '(b c) l -> b c l', b=B, c=C)  # [B, C, Pred_len]

        # Reshape from b c l to b l c
        x = rearrange(x, 'b c l -> b l c')  # [B, Pred_len, C]
        x = self.final_layer(x).squeeze()  # [B, Pred_len, c_out]
        # print('P-sLSTM output shape:', x.shape)

        return x, x
    


### implementation of DUET model

from DUET.ts_benchmark.baselines.duet.layers.linear_extractor_cluster import Linear_extractor_cluster
from DUET.ts_benchmark.baselines.duet.utils.masked_attention import Mahalanobis_mask, Encoder, EncoderLayer, FullAttention, AttentionLayer


class DUET(nn.Module):
    def __init__(self, config):
        super(DUET, self).__init__()
        self.cluster = Linear_extractor_cluster(config)
        self.CI = config.CI
        self.n_vars = config.enc_in
        self.mask_generator = Mahalanobis_mask(config.seq_len)
        self.Channel_transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model)
        )

        self.linear_head = nn.Sequential(nn.Linear(config.d_model, config.pred_len), nn.Dropout(config.fc_dropout))
        self.final_layer = nn.Linear(config.enc_in, config.c_out)

    def forward(self, input):
        # x: [batch_size, seq_len, n_vars]
        if self.CI:
            channel_independent_input = rearrange(input, 'b l n -> (b n) l 1')

            reshaped_output, L_importance = self.cluster(channel_independent_input)

            temporal_feature = rearrange(reshaped_output, '(b n) l 1 -> b l n', b=input.shape[0])

        else:
            temporal_feature, L_importance = self.cluster(input)

        # B x d_model x n_vars -> B x n_vars x d_model
        temporal_feature = rearrange(temporal_feature, 'b d n -> b n d')
        if self.n_vars > 1:
            changed_input = rearrange(input, 'b l n -> b n l')
            channel_mask = self.mask_generator(changed_input)

            channel_group_feature, attention = self.Channel_transformer(x=temporal_feature, attn_mask=channel_mask)

            output = self.linear_head(channel_group_feature)
        else:
            output = temporal_feature
            output = self.linear_head(output)

        output = rearrange(output, 'b n d -> b d n')
        output = self.cluster.revin(output, "denorm")
        output = self.final_layer(output).squeeze()
        return L_importance, output