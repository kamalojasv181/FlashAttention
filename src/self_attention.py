import torch

class CausalSelfAttention(torch.nn.Module):

    def __init__(self, config):

        super(CausalSelfAttention, self).__init__()
        
        self.c_attn = torch.nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = torch.nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = torch.nn.Dropout(config.dropout)
        self.resid_dropout = torch.nn.Dropout(config.dropout)

        self.register_buffer("mask", torch.tril(torch.ones(config.context_length, config.context_length)).view(1, 1, config.context_length, config.context_length))

        self.n_head = config.n_head

        

    def forward(self, x):

        batch_size, context_length, n_embd = x.shape

        # x: [batch_size, context_length, n_embd]
        x = self.c_attn(x) # [batch_size, context_length, n_embd * 3]
        query, key, value = x.split(x.size(-1) // 3, dim=-1) # [batch_size, context_length, n_embd]

        query = query.view(batch_size, context_length, self.n_head, n_embd // self.n_head).transpose(1, 2) # [batch_size, n_head, context_length, n_embd // n_head]

        key = key.view(batch_size, context_length, self.n_head, n_embd // self.n_head).transpose(1, 2) # [batch_size, n_head, context_length, n_embd // n_head]

        value = value.view(batch_size, context_length, self.n_head, n_embd // self.n_head).transpose(1, 2) # [batch_size, n_head, context_length, n_embd // n_head]

        attn = torch.matmul(query, key.transpose(-1, -2)) # [batch_size, n_head, context_length, context_length]
        attn = attn / (n_embd // self.n_head) ** 0.5

        mask = self.mask[:, :, :context_length, :context_length]

        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = torch.nn.Softmax(dim=-1)(attn)

        attn = self.attn_dropout(attn)

        x = torch.matmul(attn, value) # [batch_size, n_head, context_length, n_embd // n_head]

        x = x.transpose(1, 2).contiguous().view(batch_size, context_length, n_embd) # [batch_size, context_length, n_embd]

        x = self.c_proj(x) # [batch_size, context_length, n_embd]

        x = self.resid_dropout(x)

        return x