import torch
from torch import nn
from fast_transformers.attention import AttentionLayer
from fast_transformers.attention.full_attention import FullAttention

class Aggre(nn.Module):
        
    def __init__(self,smiles_reflect_dim,adduct_len=3,ecfp_len=1024,mz_len=1,dropout=0.2):
        super().__init__()    
        # 1. 对 `adduct`  'ecfp' 使用嵌入映射
        self.emd_size =1024
        self.adduct_emb = nn.Embedding(adduct_len, self.emd_size//2)
        self.ecfp_emb = nn.Linear(ecfp_len, self.emd_size)
        self.smiles_embedding_layer = nn.Linear(smiles_reflect_dim, self.emd_size)

    def forward(self, smiles_embedding ,m_z, adduct, ecfp):

        # mask   [batch, length]
        # m/z    [batch]
        # adduct [batch] 
        # ecfp   [batch, ecfp_length]

        adduct = self.adduct_emb(adduct)
        # adduct: [batch,1]

        ecfp = self.ecfp_emb(ecfp)
        # [batch_size, 1]

        m_z = m_z.unsqueeze(1).repeat(1,self.emd_size//2)
        # [batch_size, 1]

        smiles_embedding = self.smiles_embedding_layer(smiles_embedding)
        # 汇聚堆叠在一起
        
        cat = torch.cat((ecfp,smiles_embedding,adduct,m_z),dim=1)

        return cat