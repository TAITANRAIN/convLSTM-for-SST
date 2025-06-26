
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding, bias=bias)
        self.hidden_dim = hidden_dim

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        conv_out = self.conv(combined)
        (cc_i, cc_f, cc_o, cc_g) = torch.split(conv_out, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        self.conv_last = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, input_seq):
        batch_size, seq_len, _, height, width = input_seq.size()
        h = torch.zeros(batch_size, self.cell.hidden_dim, height, width, device=input_seq.device)
        c = torch.zeros(batch_size, self.cell.hidden_dim, height, width, device=input_seq.device)
        outputs = []
        for t in range(seq_len):
            h, c = self.cell(input_seq[:, t], h, c)
            outputs.append(h.unsqueeze(1))
        last_h = outputs[-1].squeeze(1)
        pred = self.conv_last(last_h)
        return pred.unsqueeze(1)