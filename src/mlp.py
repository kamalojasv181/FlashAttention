import torch

class MLP(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = torch.nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = torch.nn.GELU()
        self.c_proj  = torch.nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        return x