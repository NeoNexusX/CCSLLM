from sklearn.model_selection import train_test_split
from rdkit.Chem import rdMolDescriptors
import torch
import torch.nn as nn
from data_pre.data_loader import calculate_ecfp
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define Autoencoder
class Autoencoder(pl.LightningModule):
    def __init__(self, input_dim, latent_dim, mode='discrete'):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            # nn.Sigmoid(),
        )
        self.latent_dim =latent_dim

        # self.loss = nn.BCEWithLogitsLoss()
        self.loss = nn.BCEWithLogitsLoss() if mode == 'discrete'  else nn.MSELoss()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def training_step(self,batch,batch_idx):
        x = batch[0]
        encoder,y_hat = self(x)
        loss = self.loss(y_hat,x)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self,batch):
        x = batch[0]
        encoder,y_hat = self(x)
        loss = self.loss(y_hat,x)
        self.log('test_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss"
            }
        }



def train_predict_autodecoer(test_emb,train_emb,unit_name=''):
    
    print(f'pl version is : {pl.__version__}')
    if pl.__version__ == '1.1.5':
        pass
    else:
        torch.set_float32_matmul_precision('high')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # test_dataset = TensorDataset(test_emb)
    # train_dataset = TensorDataset(train_emb)

    all_emb = torch.cat((test_emb,train_emb),dim=0)
    all_dataset = TensorDataset(all_emb)
    
    # dataloader
    num_workers = 32 if all_emb.device == 'cpu' else 0 
    all_dataloader= DataLoader(all_dataset, batch_size=64, shuffle=False,num_workers=num_workers)
    
    autoencoder = Autoencoder(input_dim =test_emb.shape[1],latent_dim=2,mode=unit_name)

    save_dir = './autodecoder_emb'

    trainer = pl.Trainer(
        max_epochs=2,
        gpus=1,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor='train_loss',
                patience=30,
                mode='min'
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        ],
        log_every_n_steps=10,
        logger=pl.loggers.TensorBoardLogger(save_dir, name='ecfp_logs')
    )

    # fit  指定数据集
    trainer.fit(autoencoder, all_dataloader)
    # fit 指定test数据集？


    autoencoder.eval()
    autoencoder = autoencoder.to(device)

    with torch.no_grad():
        test_emb = test_emb.to(device)
        train_emb = train_emb.to(device)
        encoded_train = autoencoder.encoder(train_emb).cpu().numpy()
        encoded_test = autoencoder.encoder(test_emb).cpu().numpy()

        # 计算距离：
        distances = np.linalg.norm(encoded_test[:, np.newaxis, :] - encoded_train, axis=2)

        # 计算每个 data 到 encoded_train 所有点的平均距离
        average_distances = distances.mean(axis=1)
        avg_dist = average_distances.mean()
        print(f'average_distances :{average_distances.mean()}')

    plt.figure(figsize=(10, 7))
    plt.scatter(encoded_train[:, 0], encoded_train[:, 1], label="Train", alpha=0.6,s=10)
    plt.scatter(encoded_test[:, 0], encoded_test[:, 1], label="Test", alpha=0.6,s=10)
    plt.title("Visualization of Encoded Molecules")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")

    # Add average distance text in the upper right corner
    plt.text(0.5, 0.98, f'Average Distance: {avg_dist:.4f}', 
            transform=plt.gca().transAxes,  # Use relative coordinates
            horizontalalignment='right',     # Right-align the text
            verticalalignment='top',         # Top-align the text
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))  # Add white background

    plt.legend()
    plt.savefig('./results/'+unit_name+'_autodecoder.png')
    plt.show()