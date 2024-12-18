import pandas as pd
from data_pre import ALLCCS_PATH, TEMP_SAVE, MELTIN_PATH
from data_pre.data_module import Data_reader_METLIN

if __name__ == '__main__':

    meltin_tester = Data_reader_METLIN(MELTIN_PATH,
                                       fun=lambda path, col_name: pd.read_csv(path, usecols=col_name))

    # select specific data
    meltin_tester.selected_proprties({'Dimer.1': 'Monomer'})
    meltin_tester.can2iso_smiles_pub()
    # meltin_tester.iso2can_smiles_cir()
    meltin_tester.data.info()
    # meltin_tester.can2iso_smiles_pub()

    # write data as csv
    meltin_tester.data.to_csv(TEMP_SAVE + 'METLIN_iso.csv', index=False)