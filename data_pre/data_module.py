import time
import pandas as pd
import random
from concurrent.futures.thread import ThreadPoolExecutor
from .data_utils import tran_iupac2smiles_fun, tran_iso2can_rdkit, restful_pub_finder, \
    SMILES_BASE_FINDER, tran_iupac2can_smiles_cir, restful_pub_name_finder

def multithreader(func):
        """
        a decorator for mult itreader methods
        """

        def decorator(self, *args, **kwargs):

            data_list = []
            if self.max_workers >= len(self.data):
                self.max_workers = len(self.data)

            each_num = len(self.data) // self.max_workers

            print(f"begin multitreader function:\r\n"
                  f"self.data len is {len(self.data)}\r\n"
                  f"func is {func.__name__}\r\n"
                  f"max_workers is {self.max_workers}\r\n"
                  f"each_num is {each_num}\r\n")

            for i in range(self.max_workers):
                data_list.append(self.data.iloc[i * each_num: (i + 1) * each_num])

            # remainder to catch the last part of the data
            remainder = len(self.data) % self.max_workers

            if remainder > 0:
                print(f"{self.data.iloc[-remainder:]}")
                data_list.append(self.data.iloc[-remainder:])

            print(f"data_list len is {len(data_list)}\r\n"
                  f"remainder len is {remainder}\r\n")

            # with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            with ThreadPoolExecutor(max_workers=self.max_workers + 1) as pool:
                futures = []

                for data in data_list:
                    random_time_rest = random.randint(1, 2)
                    time.sleep(random_time_rest)
                    futures.append(pool.submit(func, self, data, *args, **kwargs))
                # futures = [pool.submit(i_fun, data) for data in data_list]
                # futures = pool.map(i_fun, data_list)

                for future in futures:
                    try:
                        result = future.result()  # 获取任务结果
                        print(f"Task result: {result}")
                    except Exception as e:
                        print(f"An error occurred: {e}")

        return decorator

# 读入部分数据集 作为dataframe 包含常规的预处理
class Data_reader:
    """
    read all kinds of database it's a basic class for you to use
    """

    def __init__(self, path_list, target_colnames, max_workers=32, fun=None):
        """
        init fun for building Data_reader class

        path_list : list each element is a string to describe a path
        target_colnames : support for pandas dataframe read specific cols
        fun : read funs must return a dataframe or using the pandas api
        """
        self.target_colnames = target_colnames
        self.path_list = path_list
        self.read_fun = fun
        self.max_workers = max_workers
        self.data = None

        # init part
        self._read_data_list()
        self._aggregate()
        self._prepeocess()
    

    def _read_data_list(self):

        try:
            # For example : data = pd.read_excel(self.xlxs_path,sheet_name=self.sheet_name)
            if hasattr(self.path_list, '__len__') and len(self.path_list) >= 1:
                self.data_list = [self.read_fun(path, self.target_colnames) for path in self.path_list]

        except IOError:
            print('data file error path is not correct :\r\n ', str(self.path_list))

        else:
            print('data read is done :')
            print(self.path_list)

    def _aggregate(self):

        if len(self.data_list) >= 1:
            self.data = pd.concat(self.data_list, axis=0, ignore_index=True)
            # 推定这两个行数相等
            assert sum([len(data) for data in self.data_list]) == self.data.shape[0]

    def _prepeocess(self):
        # 过滤
        self.data.drop_duplicates(None, keep='first', inplace=True, ignore_index=True)
        self.data.dropna(axis=0, how='any', inplace=True)
        print("finish prepeocess with data , info is:")
        self.data.info()

    def iupac2smiles(self, col_name='smiles', supply_name='Molecule Name'):

        """
        try to trans iupac to smiles 
        col_name : contain smiles column name
        """

        self.data[col_name] = self.data[supply_name].apply(tran_iupac2smiles_fun)

    def iso2can_smiles_offline(self, col_name):

        """
        try to trans can to iso smiles through rdkit
        col_name : contain smiles column name 
        """

        self.data[col_name] = self.data[col_name].apply(tran_iso2can_rdkit)

    def iso2can_smiles_cir(self, col_name='smiles'):

        """
        try to trans can to iso smiles through CDIR
        col_name : column which contain smiles  name
        """
        # transformer into iso smiles 
        self.data[col_name] = self.data[col_name].apply(tran_iupac2can_smiles_cir)

    @multithreader
    def supply_smiles(self, target_data, col_name='smiles', supply_name='Molecule Name', transformer=None):

        """
        try to fill na smiles
        col_name : column which contain smiles  name
        """
        print("supply_smiles is running")

        # default use Molecule Name to supply more information
        if transformer:
            func = lambda x: transformer(x[supply_name])
        else:
            func = lambda x: restful_pub_name_finder(x[supply_name])

        # add more smiles to fill the empty
        target_data.loc[target_data[col_name].isna(), col_name] = target_data.loc[target_data[col_name].isna()].apply(
            func, axis=1)

    def selected_proprties(self, selected):

        for key, value in selected.items():
            self.data = self.data[self.data[key] == value]
            self.data = self.data.reset_index(drop=True)
        print("finish data selection work, data info :")
        self.data.info()

    def print_uniques(self):
        for col in self.data.columns:
            print(f'col is {col} \r\n '
                  f'numbers: {len(self.data[col].unique())}\r\n'
                  f'types :{self.data[col].unique()}\r\n')
            print(f'col is {col} \r\n '
                  f'numbers: {len(self.data[col].unique())}\r\n'
                  f'types :{self.data[col].unique()}\r\n')

    def random_split(self, count):

        assert len(count) == 2
        random_numbers_test = random.sample(range(len(self.data)), count[0])
        random_numbers_valid = random.sample(range(len(self.data)), count[1])
        random_data_test = self.data.loc[random_numbers_test]
        random_data_valid = self.data.loc[random_numbers_valid]
        self.data = self.data.drop(index=random_numbers_test + random_numbers_valid)

        return random_data_test, random_data_valid

    @multithreader
    def can2iso_smiles_pub(self, target_data, col_name='smiles', supply_name='Molecule Name', transformer=None):
        print("can2iso_smiles_pub is running")
        try:
            assert col_name in target_data.columns
            assert supply_name in target_data.columns
        except AssertionError:
            print(f"can2iso_smiles_pub col_name :{col_name}  or supply_name :{supply_name} is not in target_data")

        try:
            assert col_name in target_data.columns
            assert supply_name in target_data.columns
        except AssertionError:
            print(f"can2iso_smiles_pub col_name :{col_name}  or supply_name :{supply_name} is not in target_data")

        # default use Molecule Name to supply more information
        transformer = transformer if transformer else lambda x: restful_pub_finder(x, SMILES_BASE_FINDER)

        target_data.loc[:, col_name] = target_data.loc[:, col_name].apply(transformer)

        print("finish can2iso_smiles_pub")

    def isomer_finder(self, group_name_list: list, index, save_index=False):
        """
        function to find repeated smiles and set it as None
        group_name_list ： List which column you want to group
        index ： which ID you use to identify repeated smiles， such as ALLCCSID OR METLINID
        """
    def isomer_finder(self, group_name_list: list, index, save_index=False):
        """
        function to find repeated smiles and set it as None
        group_name_list ： List which column you want to group
        index ： which ID you use to identify repeated smiles， such as ALLCCSID OR METLINID
        """
        grouped = self.data.groupby(group_name_list)[index]
        smiles_column_name = group_name_list[0]
        smiles_column_name = group_name_list[0]
        for group, data in grouped:
            if data.nunique() > 1:
                print(f"Group: {group}")
                print(data)

        # 找到有重复 index 的分组（即分组内的 index 数量大于 1）
        repeated_smiles = grouped.filter(lambda x: x.nunique() > 1)

        # choose to save data or not
        if save_index:
            repeated_smiles.to_csv('output.csv', index=False)
        if save_index:
            repeated_smiles.to_csv('output.csv', index=False)

        # deal with all isomer data
        for idx in repeated_smiles:
            if idx in self.data[index].values:
                self.data.loc[self.data[index] == idx, smiles_column_name] = None
                self.data.loc[self.data[index] == idx, smiles_column_name] = None

        # return a series include index number
        return repeated_smiles


class Data_reader_METLIN(Data_reader):
    """
    Example :

    using the Data_reader_METLIN:

    meltin_tester = Data_reader_METLIN(MELTIN_PATH,
                                        fun=lambda path, col_name: pd.read_csv(path, usecols=col_name))
    meltin_tester.isomer_finder(
        ['smiles', 'Adduct'],
        'Molecule Name',
        'Molecule Name')

    select specific data
    meltin_tester.selected_proprties({'Adduct': 1,
                                      'Dimer.1': 'Monomer'})
    write data as csv
    meltin_tester.data.to_csv(TEMP_SAVE + 'METLIN.CSV', index=False)

    Args :

    path_list : list each element is a string to describe a path
    target_colnames : support for pandas dataframe read specific cols default is  ['Molecule Name','CCS_AVG','Adduct','Dimer.1','inchi','smiles']
    fun : read funs must return a dataframe or using the pandas api

    """

    def __init__(self, path_list, target_colnames=None, max_workers=2, fun=None):
        target_colnames = target_colnames if target_colnames else ['Molecule Name', 'CCS_AVG',
                                                                   'Adduct', 'Dimer.1',
                                                                   'inchi', 'smiles', 'm/z']
        super().__init__(path_list, target_colnames, max_workers, fun)


class Data_reader_ALLCCS(Data_reader):
    """
    Example :

    allccs_tester = Data_reader_ALLCCS(ALLCCS_PATH,
                                        fun=lambda path, col_name: pd.read_csv(path, usecols=col_name))

    allccs_tester.selected_proprties({'Type': 'Experimental CCS'})

    allccs_tester.data = allccs_tester.data[allccs_tester.data['Confidence level'] != 'Conflict']

    allccs_tester.isomer_finder(
        ['Structure', 'Adduct'],
        'AllCCS ID',
        'Name')

    allccs_tester.data.to_csv(TEMP_SAVE + 'ALLCCS.csv', index=False)

    Args :

    path_list : list each element is a string to describe a path
    target_colnames : support for pandas dataframe read specific cols default is  ['Molecule Name','CCS_AVG','Adduct','Dimer.1','inchi','smiles']
    fun : read funs must return a dataframe or using the pandas api

    """

    def __init__(self, path_list, target_colnames=None, max_workers=32, fun=None):
        target_colnames = target_colnames if target_colnames else ['AllCCS ID', 'Name', 'Formula', 'Type', 'Adduct',
                                                                   'm/z', 'CCS', 'Confidence level', 'Structure']
        super().__init__(path_list, target_colnames, max_workers, fun)
