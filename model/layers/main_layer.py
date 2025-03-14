import torch
import random
import os
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
import numpy as np
from scipy.stats import pearsonr
from ..rotate_attention.rotate_builder import RotateEncoderBuilder as rotate_builder
from fast_transformers.feature_maps import GeneralizedRandomFeatures
from fast_transformers.masking import LengthMask as LM
from fast_transformers.masking import FullMask as FL
from functools import partial
from sklearn.metrics import r2_score
from .head_layer import Head
import pandas as pd
from apex import optimizers
# from .aggre_linear_layer import Aggre
# from .aggre_attention_layer import Aggre
from .aggre_layer import Aggre

def append_to_file(filename, line):
    with open(filename, "a") as f:
        f.write(line + "\n")



class LightningModule(pl.LightningModule):

    def __init__(self, config, tokenizer):
        super(LightningModule, self).__init__()
        # 读取参数然后 将参数给特定的参数
        self.config = config
        self.hparams = config
        self.mode = config.mode
        self.save_hyperparameters(config)
        self.tokenizer=tokenizer
        self.min_loss = {
            self.hparams.measure_name + "min_valid_loss": torch.finfo(torch.float32).max,
            self.hparams.measure_name + "min_epoch": 0,
        }

        self.hparams.project_name
        # Word embeddings layer
        # 字典，嵌入曾参数初始化？这里没用上
        n_vocab, d_emb = len(tokenizer.vocab), config.n_embd
        # input embedding stem
        # 核心模块构建
        builder = rotate_builder.from_kwargs(
            n_layers=config.n_layer,
            n_heads=config.n_head,
            query_dimensions=config.n_embd//config.n_head,
            value_dimensions=config.n_embd//config.n_head,
            feed_forward_dimensions=config.n_embd,
            attention_type='linear',
            feature_map=partial(GeneralizedRandomFeatures, n_dims=config.num_feats),
            activation='gelu',
            )
        # 以上为参数初始化

        # 位置编码没有
        self.pos_emb = None
        # 输入嵌入层初始化
        self.tok_emb = nn.Embedding(n_vocab, config.n_embd)
        self.drop = nn.Dropout(config.d_dropout)
        # transformer 的初始化 
        ## transformer
        self.blocks = builder.get()
        # 词翻译层初始化
        self.lang_model = self.lm_layer(config.n_embd, n_vocab)
        self.train_config = config
        #if we are starting from scratch set seeds
        #########################################
        # protein_emb_dim, smiles_embed_dim, dims=dims, dropout=0.2):
        #########################################

        self.fcs = []  

        # 默认使用了 L1 loss
        self.loss = torch.nn.MSELoss()
        # self.loss = torch.nn.L1Loss()
        # self.net = Head(config.n_embd, dims=config.dims, dropout=config.dropout)
        # self.aggre = Aggre(config.n_embd)
        
        self.aggre = Aggre(config.n_embd,adduct_len=10)
        self.net = Head(self.aggre.emd_size*3, dims=config.dims, dropout=config.dropout)
        self.max_r2 = 0
        
    class lm_layer(nn.Module):
        # lang 实现， 在训练不起任何作用，只为了预训练预测使用
        def __init__(self, n_embd, n_vocab):
            super().__init__()
            self.embed = nn.Linear(n_embd, n_embd)
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, n_vocab, bias=False)
        def forward(self, tensor):
            tensor = self.embed(tensor)
            tensor = F.gelu(tensor)
            tensor = self.ln_f(tensor)
            tensor = self.head(tensor)
            return tensor

    def get_loss(self, smiles_emb, measures):
        #  loss 计算 squeeze 移除 （N，1）
        z_pred = self.net.forward(smiles_emb).squeeze()
        # z_pred = self.net.forward(smiles_emb)
        measures = measures.float()

        return self.loss(z_pred, measures), z_pred, measures
    
    def on_save_checkpoint(self, checkpoint):
        #save RNG states each time the model and states are saved
        out_dict = dict()
        out_dict['torch_state']=torch.get_rng_state()
        out_dict['cuda_state']=torch.cuda.get_rng_state()
        if np:
            out_dict['numpy_state']=np.random.get_state()
        if random:
            out_dict['python_state']=random.getstate()
        checkpoint['rng'] = out_dict

    def on_load_checkpoint(self, checkpoint):
        #load RNG states each time the model and states are loaded from checkpoint
        rng = checkpoint['rng']
        for key, value in rng.items():
            if key =='torch_state':
                torch.set_rng_state(value)
            elif key =='cuda_state':
                torch.cuda.set_rng_state(value)
            elif key =='numpy_state':
                np.random.set_state(value)
            elif key =='python_state':
                random.setstate(value)
            else:
                print('unrecognized state')

    def _init_weights(self, module):
        
        # 初始化没有用的权重
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn == 'mix_weight':
                    decay.add(fpn)


        if self.pos_emb != None:
            no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.0},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        if self.hparams.measure_name == 'r2':
            betas = (0.9, 0.999)
        else:
            betas = (0.9, 0.999)
        print('betas are {}'.format(betas))
        learning_rate = self.train_config.lr_start * self.train_config.lr_multiplier
        optimizer = optimizers.FusedLAMB(optim_groups, lr=learning_rate, betas=betas)
        # optimizer = optim.Adam(optim_groups, lr=learning_rate, betas=betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-6) 
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "CCS_test_loss"
            }
        }

    def training_step(self, batch, batch_idx):

        # idx([8, 56])
        # mask([8, 56])
        # m/z([8])
        # adduct([8])
        # ecfp([8, 1024])
        # ccs([8])
        idx = batch[0]# idx
        mask = batch[1]# mask
        m_z = batch[2] # m/z
        adduct = batch[3] # adduct
        ecfp = batch[4] # ecfp
        targets = batch[-1] # ccs

        loss = 0

        b, t = idx.size()
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        x = self.drop(token_embeddings)
        x = self.blocks(x, length_mask=LM(mask.sum(-1)))
        token_embeddings = x
        # loss_input = self.aggre(x, m_z, adduct, ecfp,mask)
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # input_mask_expanded : [batch, seq_len, emb_dim]
        masked_embedding = token_embeddings * input_mask_expanded
        # 对 mask 进行 使用
        sum_embeddings = torch.sum(masked_embedding, 1)
        # 有效 token 的嵌入保留，无效 token 的嵌入变为 0，[batch, emb_dim]
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        loss_input = sum_embeddings / sum_mask
        loss_input = self.aggre(loss_input, m_z, adduct, ecfp)

        loss, pred, actual = self.get_loss(loss_input, targets)

        self.log('train_loss', loss, on_step=True)

        logs = {"train_loss": loss}

        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx, dataset_idx):
        idx =  val_batch[0]
        mask = val_batch[1]
        m_z = val_batch[2] # m/z
        adduct = val_batch[3] # adduct
        ecfp = val_batch[4] # ecfp
        targets = val_batch[-1]

        loss = 0

        b, t = idx.size()
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        x = self.drop(token_embeddings)
        x = self.blocks(x, length_mask=LM(mask.sum(-1)))
        # x = self.aggre(x, m_z, adduct, ecfp)
        token_embeddings = x
        # loss_input = self.aggre(x, m_z, adduct, ecfp,mask)
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        masked_embedding = token_embeddings * input_mask_expanded
        sum_embeddings = torch.sum(masked_embedding, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        loss_input = sum_embeddings / sum_mask
        loss_input = self.aggre(loss_input, m_z, adduct, ecfp)

        loss, pred, actual = self.get_loss(loss_input, targets)

        self.log('val_loss', loss, on_step=True)
        return {
            "val_loss": loss,
            "pred": pred.detach(),
            "actual": actual.detach(),
            "dataset_idx": dataset_idx,
        }
    
    def validation_epoch_end(self, outputs):
        # results_by_dataset = self.split_results_by_dataset(outputs)
        tensorboard_logs = {}
        for dataset_idx, batch_outputs in enumerate(outputs):
            dataset = self.hparams.dataset_names[dataset_idx]
            print(f"x_{dataset}_loss: {batch_outputs[0]['val_loss'].item()}")
            
            avg_loss = torch.stack([x["val_loss"] for x in batch_outputs]).mean()
            preds = torch.cat([x["pred"] for x in batch_outputs])
            actuals = torch.cat([x["actual"] for x in batch_outputs])
            val_loss = self.loss(preds, actuals)

            actuals_cpu = actuals.detach().cpu().numpy()
            preds_cpu = preds.detach().cpu().numpy()
            pearson_r = pearsonr(actuals_cpu, preds_cpu)
            r2 = r2_score(actuals_cpu, preds_cpu)

            if dataset=='test':
                if r2 > self.max_r2:
                    self.max_r2 = r2
                    # 将数据写入 CSV
                    data = {
                        'true_ccs': actuals_cpu,
                        'predicted_ccs': preds_cpu
                    }
                    df = pd.DataFrame(data)
                    r2_string = str(round(self.max_r2,2))
                    df.to_csv(self.hparams.results_dir+
                                f'/{r2_string}_'+
                                self.hparams.project_name+
                                self.hparams.dataset_name+
                                f'_{self.hparams.lr_start}_'+
                                f'{self.hparams.batch_size}_'+
                                f'{self.hparams.dropout}'+
                              '.csv', index=False)  # 保存到当前log目录的 results文件夹中

            print(f'r2 is {r2}')
            
            tensorboard_logs.update(
                {
                    # dataset + "_avg_val_loss": avg_loss,
                    self.hparams.measure_name + "_" + dataset + "_loss": val_loss,
                    self.hparams.measure_name + "_" + dataset + "_r2": r2,
                    self.hparams.measure_name + "_" + dataset + "_pearsonr": pearson_r[0],
                }
            )

        if (
            tensorboard_logs[self.hparams.measure_name + "_valid_loss"]
            < self.min_loss[self.hparams.measure_name + "min_valid_loss"]):

            self.min_loss[self.hparams.measure_name + "min_valid_loss"] = tensorboard_logs[
                self.hparams.measure_name + "_valid_loss"
            ]
            self.min_loss[self.hparams.measure_name + "min_test_loss"] = tensorboard_logs[
                self.hparams.measure_name + "_test_loss"
            ]
            self.min_loss[self.hparams.measure_name + "min_epoch"] = self.current_epoch

        tensorboard_logs[self.hparams.measure_name + "_min_valid_loss"] = self.min_loss[
            self.hparams.measure_name + "min_valid_loss"
        ]
        tensorboard_logs[self.hparams.measure_name + "_min_test_loss"] = self.min_loss[
            self.hparams.measure_name + "min_test_loss"
        ]

        self.logger.log_metrics(tensorboard_logs, self.global_step)

        learning_rate = self.optimizers().defaults['lr']
        print(f"learning_rate is {learning_rate}")

        for k in tensorboard_logs.keys():
            self.log(k, tensorboard_logs[k])

        print("Validation: Current Epoch", self.current_epoch)

        return {"avg_val_loss": avg_loss}
