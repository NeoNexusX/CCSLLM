import os
import pandas as pd
import numpy as np
from rdkit import Chem
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score
import collections
from rdkit.Chem import rdFingerprintGenerator
import csv
from pathlib import Path
from data_prepare.data_utils import calculate_ecfp_rdkit


torch.set_float32_matmul_precision('high')

class MolecularDataset(Dataset):
    def __init__(self, data_df):
        self.data_df = data_df
        self.data_df.info()
        print("Calculating ECFP features...")
        self.data_df['ecfp'] = self.data_df['smiles'].apply(calculate_ecfp_rdkit)
        self.data_df = self.data_df.dropna()

        self.fingerprints = torch.FloatTensor(self.data_df['ecfp'].tolist())
        self.mz_values = torch.FloatTensor(self.data_df['m/z'].values).reshape(-1, 1)

        counter = collections.Counter(self.data_df['Adduct'].tolist())
        sort_counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        idx2adduct = [adduct for adduct, freq in sort_counter]
        adduct2idx = {adduct: idx for idx, adduct in enumerate(idx2adduct)}
        self.data_df['Adduct'] = self.data_df['Adduct'].apply(lambda x: adduct2idx.get(x))
        adducts_tensor = torch.LongTensor(self.data_df['Adduct'])
        self.adducts = F.one_hot(adducts_tensor, num_classes=len(adduct2idx)).float()

        self.targets = torch.FloatTensor(self.data_df['CCS'].values)

        print(f"Finish preparing data:\n Adducts: {len(self.adducts[1])}")

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        features = torch.cat([
            self.fingerprints[idx],
            self.mz_values[idx],
            self.adducts[idx]
        ])
        return features, self.targets[idx]

class CCSPredictionModel(pl.LightningModule):
    def __init__(self, input_size):
        super().__init__()
        self.save_hyperparameters()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 1028),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1028, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.test_results = []  # Store results for CSV
        
    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = F.mse_loss(y_hat, y)
        r2 = r2_score(y.cpu().numpy(), y_hat.detach().cpu().numpy())
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_r2', r2, prog_bar=True)
        self.log('lr',self.optimizers().defaults['lr'])
    
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()

        #save data:
        # Collect true and predicted values
        true_values = y.cpu().numpy()
        predicted_values = y_hat.detach().cpu().numpy()

        # Store the results in a list for later saving
        for true, pred in zip(true_values, predicted_values):
            self.test_results.append({'true_ccs': true, 'predicted_ccs': pred})
            
        loss = F.mse_loss(y_hat, y)
        r2 = r2_score(y.cpu().numpy(), y_hat.detach().cpu().numpy())
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_r2', r2, prog_bar=True)
        return loss
    
    def on_test_epoch_end(self):
        # Save results to a CSV file
        output_path = Path("test_results.csv")
        with open(output_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['true_ccs', 'predicted_ccs'])
            writer.writeheader()
            writer.writerows(self.test_results)

        print(f"Test results saved to {output_path.resolve()}")


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-4
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

def train_model(train_path, val_path, test_path, save_dir='checkpoints'):
    os.makedirs(save_dir, exist_ok=True)

    # 初始化数据集和数据加载器
    train_dataset = MolecularDataset(pd.read_csv(train_path))
    val_dataset = MolecularDataset(pd.read_csv(val_path))
    test_dataset = MolecularDataset(pd.read_csv(test_path))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=4, persistent_workers=True)

    # 初始化模型
    model = CCSPredictionModel(input_size=train_dataset[0][0].shape[0])

    trainer = pl.Trainer(
        max_epochs=500,
        accelerator='gpu',
        devices=[3],
        precision=32,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                min_delta=0.01,
                mode='min'
            ),
           pl.callbacks.ModelCheckpoint(
                dirpath=save_dir,
                filename='ccs_model-{epoch:02d}-{val_loss:.2f}',
                monitor='val_loss',
                mode='min',
                save_top_k=3,
                save_last=True
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        ],
        log_every_n_steps=10,
        logger=pl.loggers.TensorBoardLogger(save_dir, name='ccs_logs')
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)

    return model, trainer

if __name__ == "__main__":
    data_path = 'data/FD_M0/'
    model, trainer = train_model(
        train_path= data_path +'FD_M0_train.csv',
        val_path= data_path + 'FD_M0_valid.csv',
        test_path= data_path + 'FD_M0_test.csv',
        save_dir='./ECFPLinear'
    )
