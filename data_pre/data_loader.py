import os
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pytorch_lightning as pl
from .tokenizer import MolTranBertTokenizer
from argparse import ArgumentParser, Namespace
from torch.utils.data import DataLoader

# adduct [M+H]=0 [M+Na]=1  [M-H]=2 
ADDUCT2IDX = {'[M+H]': 0, '[M-H]': 1, '[M+Na]': 2}

class PropertyPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, df,  measure_name, tokenizer, aug=True):
        df = df.dropna()  # TODO - Check why some rows are na
        self.df = df

        #smiles
        self.original_smiles = df["smiles"].tolist()

        #tokenizer
        self.tokenizer = MolTranBertTokenizer('bert_vocab.txt')
        self.measures = df[measure_name].tolist()  if measure_name else None

        # calculate ecfp 
        df['ecfp'] = df['smiles'].apply(calculate_ecfp)
        self.ecfp = df['ecfp'].tolist()
        df.info()

        # mz 
        self.mz = df['m/z'].tolist()

        # adduct
        self.adduct_origin = df['Adduct'].tolist()
        df['Adduct'] = df['Adduct'].apply(lambda x: ADDUCT2IDX.get(x))
        self.adduct = df['Adduct'].tolist()

        # print(f"ADDUCT2IDX:{ADDUCT2IDX}\r\noriginal_smiles:{self.original_smiles[69]}\r\necfp:{self.ecfp[69]}\r\nmz:{self.mz[69]}\r\nadduct:{self.adduct[69]}\r\nadduct:{self.adduct_origin[69]}\r\n")
        # print(f"ADDUCT2IDX:{ADDUCT2IDX}\r\noriginal_smiles:{self.original_smiles[70]}\r\necfp:{self.ecfp[70]}\r\nmz:{self.mz[70]}\r\nadduct:{self.adduct[70]}\r\nadduct:{self.adduct_origin[70]}\r\n")

    def __getitem__(self, index):
        # 每一个元素是 [smiles, measure,m/z,adduct,ecfp]
        return self.original_smiles[index], self.measures[index], self.mz[index], self.adduct[index], self.ecfp[index]

    def __len__(self):
        return len(self.original_smiles)

class PropertyPredictionDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super(PropertyPredictionDataModule, self).__init__()
        if type(hparams) is dict:
            hparams = Namespace(**hparams)
        self.hparams = hparams
        #self.smiles_emb_size = hparams.n_embd
        self.tokenizer = MolTranBertTokenizer('bert_vocab.txt')
        self.dataset_name = hparams.dataset_name

    def get_split_dataset_filename(dataset_name, split):
        return dataset_name + "_" + split + ".csv"

    def prepare_data(self):
        print("Inside prepare_dataset")
        train_filename = PropertyPredictionDataModule.get_split_dataset_filename(
            self.dataset_name, "train"
        )

        valid_filename = PropertyPredictionDataModule.get_split_dataset_filename(
            self.dataset_name, "valid"
        )

        test_filename = PropertyPredictionDataModule.get_split_dataset_filename(
            self.dataset_name, "test"
        )

        train_ds = get_dataset(
            self.hparams.data_root,
            train_filename,
            self.hparams.train_dataset_length,
            self.hparams.aug,
            measure_name=self.hparams.measure_name,
        )

        val_ds = get_dataset(
            self.hparams.data_root,
            valid_filename,
            self.hparams.eval_dataset_length,
            aug=False,
            measure_name=self.hparams.measure_name,
        )

        test_ds = get_dataset(
            self.hparams.data_root,
            test_filename,
            self.hparams.eval_dataset_length,
            aug=False,
            measure_name=self.hparams.measure_name,
        )

        self.train_ds = train_ds
        self.val_ds = [val_ds] + [test_ds]
        # 合并成一个 list【val_ds，test_ds】 方便后边迭代
        # print(
        #     f"Train dataset size: {len(self.train_ds)}, val: {len(self.val_ds1), len(self.val_ds2)}, test: {len(self.test_ds)}"
        # )

    def collate(self, batch):
        # 将每一个分子式smiles 编码 tokenlize， 同时执行长度填充，动态填充到每一个批次大小相同
        tokens = self.tokenizer.batch_encode_plus([ smile[0] for smile in batch ], padding=True, add_special_tokens=True)
        # 返回的是【tokens，mask，ans】 原来是这样的 现在是下面数据
        # 每一个元素是 [smiles, measure,m/z,adduct,ecfp]
        return (torch.tensor(tokens['input_ids']), # input
                torch.tensor(tokens['attention_mask']), # mask
                torch.tensor([smile[2] for smile in batch]),# m/z
                torch.tensor([smile[3] for smile in batch]),# adduct
                torch.tensor([smile[4] for smile in batch],dtype=torch.float32),# ecfp
                torch.tensor([smile[1] for smile in batch]))# ccs

    def val_dataloader(self):
        return [
            DataLoader(
                ds,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                shuffle=False,
                collate_fn=self.collate,
            )
            for ds in self.val_ds
        ]

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=self.collate,
        )

class CheckpointEveryNSteps(pl.Callback):
    """
        Save a checkpoint every N steps, instead of Lightning's default that checkpoints
        based on validation loss.
    """

    def __init__(self, save_step_frequency=-1,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
        ):
        """
        Args:
        save_step_frequency: how often to save in steps
        prefix: add a prefix to the name, only used if
        use_modelcheckpoint_filename=False
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        if global_step % self.save_step_frequency == 0 and self.save_step_frequency > 10:

            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch}_{global_step}.ckpt"
                #filename = f"{self.prefix}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)

class ModelCheckpointAtEpochEnd(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        metrics['epoch'] = trainer.current_epoch
        if trainer.disable_validation:
            trainer.checkpoint_callback.on_validation_end(trainer, pl_module)
            
def get_dataset(data_root, filename, dataset_len, aug, measure_name):
    df = pd.read_csv(os.path.join(data_root, filename))
    print("Length of dataset:", len(df))
    if dataset_len:
        df = df.head(dataset_len)
        print("Warning entire dataset not used:", len(df))
    dataset = PropertyPredictionDataset(df,  measure_name, aug)
    return dataset

def calculate_ecfp(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:  # 如果SMILES字符串有效
        ecfp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return list(ecfp) # 转换为list数组
    else:
        return None  # 如果SMILES无效，则返回一个全0的指纹


