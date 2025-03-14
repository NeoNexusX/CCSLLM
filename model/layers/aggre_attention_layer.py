import torch
from torch import nn
import torch.nn.functional as F
from fast_transformers.attention import AttentionLayer
from fast_transformers.attention.full_attention import FullAttention
from fast_transformers.masking import LengthMask as LM
from fast_transformers.masking import FullMask
from yarl import Query
from model.rotate_attention import attention_layer

class Aggre(nn.Module):
    def __init__(self, 
                 smiles_embed_dim, 
                 ecfp_length=1024,
                 num_adducts=3,  # 根据实际离子类型数量调整
                 dropout=0.1):
        
        super().__init__()
        
        mz_dim = smiles_embed_dim
        adduct_dim = smiles_embed_dim
        ecfp_dim = smiles_embed_dim

        # SMILES特征增强
        self.smiles_attn = nn.Sequential(
            nn.Linear(smiles_embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # 数值特征处理
        self.mz_net = nn.Sequential(
            nn.Linear(1, mz_dim),
            nn.LayerNorm(mz_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 离子类型嵌入
        self.adduct_emb = nn.Embedding(num_adducts, adduct_dim)
        self.adduct_net = nn.Sequential(
            nn.Linear(adduct_dim, adduct_dim),
            nn.LayerNorm(adduct_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # ECFP特征处理
        self.ecfp_net = nn.Sequential(
            nn.Linear(ecfp_length, ecfp_dim),
            nn.LayerNorm(ecfp_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 跨模态融合
        heads_num = 12 
        self.cross_attn = AttentionLayer(
            attention=FullAttention(),  # 使用线性注意力
            d_model=smiles_embed_dim,      # 输入维度
            n_heads=heads_num,                     # 头数保持与原来一致
            d_keys=smiles_embed_dim//heads_num,    # 键维度
            d_values=smiles_embed_dim//heads_num,  # 值维度
            event_dispatcher=None
        )
        
        # 最终融合层
        total_dim = smiles_embed_dim * 4
        self.final_fusion = nn.Sequential(
            nn.Linear(total_dim, total_dim//2),
            nn.LayerNorm(total_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(total_dim//2, smiles_embed_dim),
            nn.LayerNorm(smiles_embed_dim)
        )
    
    def forward(self, smiles_emb, m_z, adduct, ecfp,mask):

        # smiles_emb   [batch, length, embsize]
        # mask   [batch, length]
        # m/z    [batch]
        # adduct [batch]
        # ecfp   [batch, ecfp_length]


        attn_weights = self.smiles_attn(smiles_emb).squeeze(-1)  # [batch, seq_len]
        attn_weights = attn_weights.masked_fill(~mask.bool(), -1e9)
        attn_weights = F.softmax(attn_weights, dim=1)
        smiles_feat = torch.einsum('bsd,bs->bd', smiles_emb, attn_weights)  # [batch, embed_dim]

        # 数值特征处理
        mz_feat = self.mz_net(m_z.unsqueeze(-1))  # [batch, mz_dim]
        
        # 离子类型特征
        adduct_feat = self.adduct_net(self.adduct_emb(adduct))  # [batch, adduct_dim]
        
        # ECFP特征
        ecfp_feat = self.ecfp_net(ecfp)  # [batch, ecfp_dim]
        
        # 跨模态注意力（使用数值特征作为query）
        cross_query = torch.cat([
            mz_feat.unsqueeze(1), 
            adduct_feat.unsqueeze(1),
            ecfp_feat.unsqueeze(1)
        ], dim=1)  # [batch, 3, embed_dim]
        
        length_mask=LM(mask.sum(-1))
        attention_mask = FullMask(cross_query.shape[1])
        query_mask =  LM(torch.full((cross_query.shape[0],), 3))

        # 注意力机制增强
        attn_out = self.cross_attn(
            queries=cross_query,
            keys= smiles_emb,
            values= smiles_emb,
            attn_mask = attention_mask,
            query_lengths = query_mask,
            key_lengths = length_mask
        )
        attn_out = attn_out.mean(dim=1)  # [batch, embed_dim]
        # attn_out = attn_out.view(attn_out.shape[0], -1)  # [batch, 3*embed_dim]

        enhanced_smiles = smiles_feat + attn_out
        combined = torch.cat([
            enhanced_smiles,
            mz_feat,
            adduct_feat,
            ecfp_feat
        ], dim=1)
        
        # 最终融合
        return self.final_fusion(combined)