import shap
import torch
import yaml
from argparse import Namespace
import numpy as np
import pandas as pd
from data_pre.data_loader import PropertyPredictionDataModule
from data_pre.tokenizer import MolTranBertTokenizer
from model.layers.main_layer import LightningModule
from fast_transformers.masking import LengthMask as LM
import matplotlib.pyplot as plt
def prepare_data(model_name):

     with open(f'Pretrained MoLFormer/hparams/{model_name}.yaml', 'r') as f:
        config = Namespace(**yaml.safe_load(f))
        
        # prepare data:
        data_module = PropertyPredictionDataModule(config)
        data_module.prepare_data()

        # data all loader
        train_loader = data_module.train_dataloader()
        test_loader = data_module.val_dataloader()[1]

        return test_loader,train_loader

def prepare_model(model_name):
        
        with open(f'Pretrained MoLFormer/hparams/{model_name}.yaml', 'r') as f:
        
            config = Namespace(**yaml.safe_load(f))

            tokenizer = MolTranBertTokenizer('bert_vocab.txt')

            ckpt = f'Pretrained MoLFormer/checkpoints/{model_name}.ckpt'

            model = LightningModule(config, tokenizer).load_from_checkpoint(ckpt, strict=False,config=config, tokenizer=tokenizer,vocab=len(tokenizer.vocab))

            # Check for GPU availability
            device = torch.device('cuda')
            model = model.to(device)  # Move model to GPU if available
            model.eval()
            
            return model
        
idx_size = 70
mask_size = 70 + idx_size
mz_size = 1 + mask_size
adduct_size = 1 + mz_size
ecfp_size =1024 + adduct_size

model_name = 'XL_87'
model = prepare_model(model_name)
# model.eval()


def predict(data):
    with torch.no_grad():
        device = 'cuda'
        
        # 确保 data 是一个 numpy 数组，然后将其转化为 torch 张量
        print("predict is running")
        data = torch.tensor(data, dtype=torch.float32).to(device)

        idx = data[:,:idx_size].long()
        mask = data[:,idx_size:mask_size]
        m_z = data[:,mask_size:mz_size].squeeze(-1)
        adduct = data[:,mz_size:adduct_size].squeeze(-1).long()
        ecfp = data[:,adduct_size:ecfp_size]
        # idx, mask, m_z, adduct, ecfp,_ = [x.to(device) for x in data]

        x = model.tok_emb(idx)
        x = model.blocks(x)
        # x = model.aggre(x, m_z, adduct, ecfp)

        input_mask_expanded = mask.unsqueeze(-1).expand(x.size()).float()
        masked_embedding = x * input_mask_expanded
        sum_embeddings = torch.sum(masked_embedding, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-8)
        loss_input = sum_embeddings / sum_mask
        loss_input = model.aggre(loss_input, m_z, adduct, ecfp)   
        pred = model.net(loss_input)

    return pred.cpu().detach().numpy()


if __name__ == '__main__':
    test_dataloader,_ = prepare_data(model_name)
    # c=
    device = 'cuda'
    first_batch = next(iter(test_dataloader))

    first_batch_processed = []
    for x in first_batch:
        if len(x.shape) == 1:  # 如果是一个一维张量，扩展成二维
            x = x.unsqueeze(-1)
        first_batch_processed.append(x.cpu().numpy())
    
    # 将所有张量堆叠在一起，确保它们有相同的形状
    first_batch_np = np.concatenate(first_batch_processed, axis=1)  # 按列（特征）进行堆叠

    data = torch.tensor(first_batch_np, dtype=torch.float32)

    # 创建一个可解释器
    explainer = shap.DeepExplainer(model, data)

    # 计算SHAP值
    shap_values = explainer.shap_values(data[0:2])

    # plot the feature attributions
    shap.image_plot(shap_values, -first_batch_np[0:2])