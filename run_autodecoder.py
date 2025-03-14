
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from model.autodecoder import Autoencoder,train_predict_autodecoer
from rdkit.Chem import rdFingerprintGenerator
from data_pre.data_utils import calculate_ecfp_rdkit
from llm_predict import prepare_llm_data


def prepare_ecfp_data():

    # Step 1: Load CSV and compute ECFP
    train_csv_file = "data/5/FINAL_DATA_train.csv"  # Replace with your file path
    train_data = pd.read_csv(train_csv_file)

    test_csv_file = "data/5/FINAL_DATA_test.csv"  # Replace with your file path
    test_data = pd.read_csv(test_csv_file)


    print("calculating ecfps")
    train_data['ecfp'] = train_data['smiles'].apply(lambda x: calculate_ecfp_rdkit(x))
    train_data = train_data.dropna(subset=['ecfp'])
    train_ecfs =  torch.FloatTensor(np.array([list(fp) for fp in train_data['ecfp']]))

    test_data['ecfp'] = test_data['smiles'].apply(lambda x: calculate_ecfp_rdkit(x))
    test_data = test_data.dropna(subset=['ecfp'])
    test_ecfs = torch.FloatTensor(np.array([list(fp) for fp in test_data['ecfp']]))

    train_data.info()
    test_data.info()

    return train_ecfs,test_ecfs

if __name__ == '__main__':

    # prepare model:
    model_name = 'XL_88'
    
    # train_ecfs,test_ecfs = prepare_ecfp_data()
    train_ecfs,test_ecfs,_,_,_,_ = prepare_llm_data(model_name)


    train_predict_autodecoer(test_ecfs,train_ecfs,unit_name='llm')