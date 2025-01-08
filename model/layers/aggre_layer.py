import torch
from torch import nn
from fast_transformers.attention import AttentionLayer
from fast_transformers.attention.full_attention import FullAttention

class Aggre(nn.Module):
        
    def __init__(self, smiles_reflect_dim,adduct_len=2,ecfp_len=1024,mz_len=1,dropout=0.2):
        super().__init__()    
        # 1. 对 `adduct`  'ecfp' 使用嵌入映射
        self.aggregator = nn.Linear(ecfp_len+adduct_len+mz_len, smiles_reflect_dim)
        self.adduct_emb = nn.Embedding(3, adduct_len)

    def forward(self, smiles_emb, m_z, adduct, ecfp, mask):

        # data   [batch, length, embsize]
        # mask   [batch, length]
        # m/z    [batch]
        # adduct [batch]
        # ecfp   [batch, ecfp_length]

        adduct = self.adduct_emb(adduct)
        # adduct: [batch,2]
        m_z = m_z.unsqueeze(1)
        # 汇聚堆叠在一起
        cat = torch.cat((ecfp,adduct,m_z),dim=1)

        # ecfp_adduct   [batch, ecfp_length + 1+]
        aggregated_output = self.aggregator(cat)

        return aggregated_output