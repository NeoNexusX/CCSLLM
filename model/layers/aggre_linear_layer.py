import torch
from torch import nn
from fast_transformers.attention import AttentionLayer
from fast_transformers.attention.full_attention import FullAttention

class Aggre(nn.Module):
        
    def __init__(self, smiles_reflect_dim,adduct_len=2,ecfp_len=1024,mz_len=1,dropout=0.2):
        super().__init__()    

        #1. to `adduct`  'ecfp'  emb refletion
        self.adduct_emb = nn.Embedding(3, adduct_len)
        self.ecfp_emb = nn.Linear(ecfp_len, smiles_reflect_dim)

        #2. Processing continuous variables
        self.mz_fc = nn.Linear(mz_len, smiles_reflect_dim)

        self.dropout = nn.Dropout(dropout)

        self.aggregator = nn.Linear(2306, smiles_reflect_dim)

    def forward(self, smiles_emb, m_z, adduct, ecfp):

        # smiles_emb   [batch, length, embsize]
        # mask   [batch, length]
        # m/z    [batch]
        # adduct [batch]
        # ecfp   [batch, ecfp_length]
        adduct = self.adduct_emb(adduct)
        adduct_emb = adduct.unsqueeze(1).repeat(1, smiles_emb.size(1), 1)

        # Step 2: Process `ecfp`
        ecfp_emb = self.ecfp_emb(ecfp)  # Shape: [batch, smiles_emb_dim]
        ecfp_emb = ecfp_emb.unsqueeze(1).repeat(1, smiles_emb.size(1), 1)  

        # Step 3: Process `m/z`
        mz_emb = self.mz_fc(m_z.unsqueeze(-1))  # Shape: [batch, smiles_emb_dim]
        mz_emb = mz_emb.unsqueeze(1).repeat(1, smiles_emb.size(1), 1)  # Broadcast to [batch, seq_len, emb_dim]


        # 汇聚堆叠在一起
        combined_features = torch.cat((smiles_emb,ecfp_emb,mz_emb,adduct_emb),dim=-1)

        # ecfp_adduct   [batch, ecfp_length + 1+]
        aggregated_output = self.aggregator(combined_features)


        return aggregated_output