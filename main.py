import pandas as pd
from data_pre import ALLCCS_PATH, TEMP_SAVE, MELTIN_PATH, BASE_PATH
from data_pre.data_module import Data_reader_METLIN, Data_reader_ALLCCS
from data_pre.data_utils import restful_pub_finder

if __name__ == '__main__':
    meltin_tester = Data_reader_METLIN(MELTIN_PATH,
                                       fun=lambda path, col_name: pd.read_csv(path, usecols=col_name))
    meltin_tester.data.info()
    meltin_tester.can2iso_smiles_pub(supply_name='Molecule Name', col_name='smiles')
    # select specific data
    # meltin_tester.selected_proprties({'Dimer.1': 'Monomer'})

    # allccs_tester = Data_reader_ALLCCS(ALLCCS_PATH,
    #                                    fun=lambda path, col_name: pd.read_csv(path, usecols=col_name))
    # select specific data
    # allccs_tester.can2iso_smiles_pub(supply_name='Name', col_name='Structure')
    # allccs_tester.supply_smiles(supply_name='Name', col_name='Structure')
    # meltin_tester.iso2can_smiles_cir()
    # allccs_tester.data.info()
    # allccs_tester.data.dropna(axis=0, how='any', inplace=True)
    # meltin_tester.can2iso_smiles_pub()
    # write data as csv
    meltin_tester.data.to_csv(BASE_PATH + 'ALLCCS_ISO.csv', index=False)



# transformer = lambda x: restful_pub_finder(x[supply_name])
# self.supply_smiles(col_name=smiles_column_name, supply_name=supply_name, transformer=transformer)
#
# self.data.drop_duplicates(group_name_list, keep='first', inplace=True, ignore_index=True)
# self.data.dropna(axis=0, how='any', inplace=True)
# self.data.info()
#
# transformer = lambda x: restful_pub_finder(x[supply_name], NAME_BASE_FINDER)
#
# self.supply_smiles(target_data, col_name=col_name, supply_name=supply_name, transformer=transformer)
