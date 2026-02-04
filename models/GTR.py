import torch
import torch.nn as nn
import torch.nn.functional as F


class GTR(nn.Module):
    def __init__(self, d_series, c, CI=False, period_len=24):
        
        super(GTR, self).__init__()
        self.agg = False
        self.period_len = period_len
        self.c = c
        self.linear = nn.Linear(d_series, d_series)
        self.CI = CI
        if self.CI:
            self.ds_convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 1 + 2 * (self.period_len // 2)),
                                stride=1, padding=(0, self.period_len // 2), padding_mode="zeros", bias=False)
            for _ in range(self.c)]
        )
        else:
            self.conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 1 + 2 * (self.period_len // 2)),
                                stride=1, padding=(0, self.period_len // 2), padding_mode="zeros", bias=False)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, q):
        _, C, S = x.shape
        # Step 1: Mapping
        global_query = self.linear(q)  # (B, C, S)

        # Step 2: GTA mode, aggregate across channels, design for capturing inter-varaible dependencies.
        if self.agg:
            weight = F.softmax(global_query, dim=1)
            global_query = torch.sum(global_query * weight, dim=1, keepdim=True)
            global_query = global_query.repeat(1, C, 1)  # (B, C, S)

        # Step 3: Fuse
        out = torch.stack([x, global_query], dim=2) # (B, C, 2, S)

        if self.CI:
            conv_outs = [
                self.ds_convs[i](out[:,i,:,:].unsqueeze(1)) # (B, 1, 2, S)
                for i in range(self.c)
                ]
            conv_out = torch.cat(conv_outs, dim=1) # (B, C, 1, S)
            conv_out = conv_out.squeeze(2)  # (B, C, S)

        else:
            out = out.reshape(-1, 1, 2, S)  # (B*C, 1, 2, S)
            conv_out = self.conv2d(out)  # (B*C, 1, 1, S)
            conv_out = conv_out.reshape(-1, C, S)  # (B, C, S)

        return self.dropout(conv_out)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_revin = configs.use_revin
        self.individual = configs.individual

        self.Q = nn.Parameter(torch.zeros(self.cycle_len, self.enc_in), requires_grad=True)
        self.GTR = GTR(d_series=self.seq_len, c=self.enc_in, CI=self.individual)
        self.input_proj = nn.Linear(self.seq_len, self.d_model)

        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )

        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len)
        )


    def forward(self, x, cycle_index):
        # RevIN normalize
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # (B, S, C) -> (B, C, S)
        x_input = x.permute(0, 2, 1)

        # GTR part
        gather_index = (cycle_index.view(-1, 1) +
                        torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)) % self.cycle_len
        query_input = self.Q[gather_index].permute(0, 2, 1)
        global_information = self.GTR(x_input, query_input)

        # Projection + MLP
        input_proj = self.input_proj(x_input + global_information)
        hidden = self.model(input_proj)
        output = self.output_proj(hidden + input_proj).permute(0, 2, 1)

        # RevIN de-normalize
        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean

        return output


