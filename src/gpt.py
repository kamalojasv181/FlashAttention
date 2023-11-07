import torch
from self_attention import CausalSelfAttention
from mlp import MLP

class Block(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = torch.nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x
    

class GPT(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

        self.wte = torch.nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = torch.nn.Embedding(config.context_length, config.n_embd)
        self.drop = torch.nn.Dropout(config.dropout)
        self.blocks = torch.nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = torch.nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, ids, targets=None):

        b, t = ids.size()
        token_embeddings = self.wte(ids)
        position_embeddings = self.wpe(torch.arange(t, device=ids.device))[None, :, :].expand(b, t, -1)

        x = self.drop(token_embeddings + position_embeddings)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss