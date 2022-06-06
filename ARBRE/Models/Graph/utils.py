import torch.nn as nn
import dgl


class Aggregator(nn.Module):

    def __init__(self, dropout):
        super(Aggregator, self).__init__()
        self.dropout = dropout
        self.message_dropout = nn.Dropout(dropout)

    def forward(self, mode, g, entity_embed):
        g = g.local_var()
        g.ndata['node'] = entity_embed
        g.update_all(dgl.function.copy_u('node', 'side'), dgl.function.sum('side', 'H'))

        neighbors = g.ndata['H']
        out = neighbors

        out = self.message_dropout(out)
        return out
