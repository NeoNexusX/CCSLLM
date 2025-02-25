from torch import nn

class Head(nn.Module): 

    # 输出层，在本层上进行输出信息调节
    def __init__(self, smiles_embed_dim, dims=None, dropout=0.1):
        super().__init__()
        self.desc_skip_connection = True 
        self.fcs = []  # nn.ModuleList()
        print('Net layer dropout is {}'.format(dropout))

        self.fc1 = nn.Linear(smiles_embed_dim, smiles_embed_dim//2)
        self.dropout1 = nn.Dropout(dropout)
        self.relu1 = nn.GELU()
        self.fc2 = nn.Linear(smiles_embed_dim//2,  smiles_embed_dim//4)
        self.relu2 = nn.GELU()
        self.final = nn.Linear(smiles_embed_dim//4, 32)
        self.relu3 = nn.GELU()
        self.final1 = nn.Linear(32, 1)


    def forward(self, smiles_emb):
    
        # # 原来的 reflict 映射层
        # x_out = self.fc1(smiles_emb)
        # x_out = self.dropout1(x_out)
        # x_out = self.relu1(x_out)

        # if self.desc_skip_connection is True:
        #     x_out = x_out + smiles_emb

        # z = self.fc2(x_out)
        # z = self.dropout2(z)
        # z = self.relu2(z)
        # if self.desc_skip_connection is True:
        #     z = self.final(z + x_out)
        # else:
        #     z = self.final(z)

        # z = self.final1(z)
        # z = self.relu3(z)
        # z = self.final2(z)

        x_out = self.fc1(smiles_emb)
        x_out = self.relu1(x_out)
        x_out = self.dropout1(x_out)
        z = self.fc2(x_out)
        z = self.relu2(z)
        z = self.final(z)
        z = self.relu3(z)
        z = self.final1(z)

        return z