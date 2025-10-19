import torch, torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(self, d_model=256, nheads=8, dec_layers=3, dim_feedforward=1024, num_queries=100, dropout=0.1):
        super().__init__()
        self.queries = nn.Embedding(num_queries, d_model)
        layer = nn.TransformerDecoderLayer(d_model, nheads, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, dec_layers)

    def forward(self, memory):  # memory: [B,HW,C]
        B, HW, C = memory.shape
        q = self.queries.weight.unsqueeze(0).repeat(B,1,1)  # [B,Nq,C]
        out = self.decoder(q, memory)  # [B,Nq,C]
        return out