from data_pre import data_module

BASE_PATH = './test_data'
TEMP_SAVE = './'
ALLCCS_PATH = [BASE_PATH + r"/AllCCS download" + rf' ({i}).csv' for i in range(43)]
MELTIN_PATH = [BASE_PATH + '/METLIN_IMS_rmTM.csv']