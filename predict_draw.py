import yaml
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from fast_transformers.masking import LengthMask as LM
from argparse import Namespace
from data_pre.tokenizer import MolTranBertTokenizer
from model.layers.main_layer import LightningModule
from data_pre.data_loader import PropertyPredictionDataModule
from view.draw import plot_ccs_comparison
from model.autodecoder import Autoencoder,train_predict_autodecoer


def predict (data_loader,model,predictions,truths,smiles_list,mz_list,tokenizer,device='cuda'):

    with torch.no_grad():
        emb_output = []
        for batch in data_loader:

            idx, mask, m_z, adduct, ecfp, true = [x.to(device) for x in batch]
            
            # Get SMILES and m/z (processed on CPU for compatibility with dataset)
            decoded_smiles_batch = [
                tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tokens))
                for tokens in idx.cpu().tolist()]
            
            smiles_list.extend(decoded_smiles_batch)
            mz_list.extend(m_z.cpu().tolist())

            # 模型前向传播
            token_embeddings = model.tok_emb(idx)
            x = model.drop(token_embeddings)
            x = model.blocks(x)
            # x = model.aggre(x, m_z, adduct, ecfp)

            # Mask处理
            input_mask_expanded = mask.unsqueeze(-1).expand(x.size()).float()
            masked_embedding = x * input_mask_expanded
            sum_embeddings = torch.sum(masked_embedding, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            loss_input = sum_embeddings / sum_mask
            emb_output.extend(loss_input)
            # loss_input = model.aggre(loss_input, m_z, adduct, ecfp)      

            #  获取预测值
            # pred = model.net(loss_input).squeeze()
            # predictions.extend(pred.cpu().tolist())
            # truths.extend(true.cpu().tolist())
        # 将列表转换为二维张量
        return torch.stack(emb_output)


def prepare_llm_data(model_name):

    with open(f'Pretrained MoLFormer/hparams/{model_name}.yaml', 'r') as f:
        config = Namespace(**yaml.safe_load(f))

        tokenizer = MolTranBertTokenizer('bert_vocab.txt')

        ckpt = f'Pretrained MoLFormer/checkpoints/{model_name}.ckpt'

        model = LightningModule(config, tokenizer).load_from_checkpoint(ckpt, strict=False,config=config, tokenizer=tokenizer,vocab=len(tokenizer.vocab))

        # Check for GPU availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)  # Move model to GPU if available
        model.eval()

        # prepare data:
        data_module = PropertyPredictionDataModule(config)
        data_module.prepare_data()

        # data all loader
        train_loader = data_module.train_dataloader()
        test_loader = data_module.val_dataloader()[1]

        #prediction
        predictions = []
        truths = []
        smiles_list = []
        mz_list = []

        # model prediction 
        print("LLM model prediction ")
        test_emb = predict(test_loader,model,predictions,truths,smiles_list,mz_list,tokenizer,device)
        print("finish test part predict")
        train_emb = predict(train_loader,model,predictions,truths,smiles_list,mz_list,tokenizer,device)
        print("finish train part predict")

        return test_emb,train_emb,predictions,truths,smiles_list,mz_list


if __name__ == '__main__':

    # prepare model:
    model_name = 'XL_87'

    test_emb,train_emb,predictions,truths,smiles_list,mz_list = prepare_llm_data(model_name)
    train_predict_autodecoer(test_emb,train_emb,unit_name = 'continuous')


   # Create and save results
    results_df = pd.DataFrame({
        'smiles': smiles_list,
        'm/z': mz_list,
        'true_ccs': truths,
        'predicted_ccs': predictions           
    })

    # 加载对比的 DataFrame
    comparison_df = pd.read_csv('data/5/ISO_METLIN_test.csv')  # 需要比较的文件

    # 确保两者的行数相等
    if len(results_df) != len(comparison_df):
        raise ValueError("not can be used col num is not same")
    
    # 获取不匹配的行
    mismatched_smiles = results_df['smiles'] != comparison_df['smiles']
    mismatched_rows = comparison_df[mismatched_smiles]
    mismatched_results_df_rows = results_df[mismatched_smiles]

    # 输出不匹配的 SMILES
    mismatched_smiles_list = mismatched_rows['smiles'].tolist()
    mismatched_results_df_rows_list = mismatched_results_df_rows['smiles'].tolist()
    print("mismatch smiles:\r\n")
    for idx, smiles in enumerate(mismatched_smiles_list, start=1):
        print(f"{idx}: {smiles}")
    
    # 保存数据结果到csv
    results_df.to_csv(f'results_{model_name}.csv', index=False)
    print(f"Results saved to results_{model_name}.csv")

    try:
        plot_ccs_comparison(results_df,f'results_{model_name}.png')
    except Exception as e:
        print(f"error : {e}")