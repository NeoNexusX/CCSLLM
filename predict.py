import torch
import yaml
from argparse import Namespace
import numpy as np
from data_pre.data_loader import PropertyPredictionDataModule
from data_pre.tokenizer import MolTranBertTokenizer
from model.layers.main_layer import LightningModule
from fast_transformers.masking import LengthMask as LM
import matplotlib.pyplot as plt
from sklearn import linear_model

def prepare_data(model_name):

     with open(f'Pretrained MoLFormer/hparams/{model_name}.yaml', 'r') as f:
        config = Namespace(**yaml.safe_load(f))
        
        # prepare data:
        data_module = PropertyPredictionDataModule(config)
        data_module.test_setup()
        data_module.prepare_data()

        # data loader
        test_loader = data_module.val_dataloader()

        return test_loader[0]

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

def predict(batch):
    with torch.no_grad():

        idx = batch[0]# idx
        mask = batch[1]# mask
        m_z = batch[2] # m/z
        adduct = batch[3] # adduct
        ecfp = batch[4] # ecfp
        targets = batch[-1] # ccs
        device = "cuda"
        idx, mask, m_z, adduct, ecfp,targets = [x.to(device) for x in batch]
        
        token_embeddings = model.tok_emb(idx) # each index maps to a (learnable) vector
        x = model.drop(token_embeddings)
        x = model.blocks(x, length_mask=LM(mask.sum(-1)))
        token_embeddings = x
        _ ,loss_input = model.aggre(x, m_z, adduct, ecfp,mask)
        pred, actual = model.get_pred(loss_input, targets)

    return pred.cpu().detach().numpy(),actual.cpu().detach().numpy()


if __name__ == '__main__':

    model_name = 'ATT_FULL'
    model = prepare_model(model_name)
    model.eval()
    test_dataloader = prepare_data(model_name)

    pre_output = []
    truth = []

    for batch in test_dataloader:
        pred,actual = predict(batch)
        pre_output.extend([pred])
        truth.extend(actual)

    y_hat = np.stack(pre_output)
    y = np.stack(truth)

    print(y_hat)
    print(y)