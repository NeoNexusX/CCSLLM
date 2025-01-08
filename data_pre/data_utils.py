import threading
import time
import random
from urllib.request import urlopen
from urllib.parse import quote
import urllib.parse
import requests
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, CanonSmiles

# URL
BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"

# BASE_FINDER
NAME_BASE_FINDER = "/name/property"
SMILES_BASE_FINDER = "/smiles/property"
INCHI_BASE_FINDER = "/inchi/property"

# SMILES_RESULT
ISOMERIC_SMILES_RESULT = "/IsomericSMILES"

# QUERY_ARG
SMILES_QUERY_ARG = "?smiles="
NAME_QUERY_ARG = "?name="
INCHI_QUERY_ARG = "?inchi="

# TXT_END
URL_TXT_END = "/TXT"

proxies = {
    "http": "http://localhost:7890",  # 将 "localhost" 和端口替换为你的代理服务器地址
    "https": "http://localhost:7890"
}

# CAN OR ISO
CANONSMILES = 1
ISOMERICSMILES = 2


def restful_pub_inchi_finder(inchi):
    return restful_pub_finder(inchi, INCHI_BASE_FINDER, INCHI_QUERY_ARG)


def restful_pub_name_finder(name):
    return restful_pub_finder(name)


def restful_pub_smiles_finder(smiles):
    return restful_pub_finder(smiles, quary_arg=SMILES_QUERY_ARG)


def restful_pub_finder(query, query_base=NAME_BASE_FINDER, quary_arg=SMILES_QUERY_ARG):
    """
    used for converting canonical_smiles to smiles through pubchem,speed : slowly
    If not find return None else return String
    
    canonical_smiles : String
    """
    if query:
        # 将Smiles名称进行URL编码
        query_arg_url = urllib.parse.quote(query, safe='')
        # 构建URL
        url = BASE_URL + query_base + ISOMERIC_SMILES_RESULT + URL_TXT_END + quary_arg + f"{query_arg_url}"
        if query_arg_url:
            try_times = 10
            for i in range(try_times):
                random_time_rest = random.randint(1, 2)
                time.sleep(random_time_rest)
                try:
                    # 发起GET请求
                    response = requests.get(url, proxies=proxies)


                    # 如果请求成功且返回200
                    if response.status_code == 200:
                        # 处理响应内容，提取SMILES编码
                        smiles_pbc = response.text.strip()

                        # 如果返回数据为空，表示没有对应数据
                        if not smiles_pbc:
                            print(f"No SMILES data found for {query}")
                            return None

                        # 返回SMILES编码
                        if '\n' in smiles_pbc:
                            smiles_pbc = None

                        print(f"{query}\r\n"
                              f"smiles_pbc find :{smiles_pbc}\r\n")
                        return smiles_pbc

                    # 如果返回的状态码不是200，打印错误的smiles编码，线程休息1s
                    else:
                        print(f"Error: {response.status_code} "
                              f"Failed to retrieve data for {query}")
                        time.sleep(random_time_rest)

                except requests.RequestException as e:
                    # 捕获网络请求异常，打印错误信息并等待1秒重试
                    print(f"Network error: {e}. Retrying in 1 seconds...")
                    time.sleep(random_time_rest)

        return None


def tran_iupac2can_smiles_cir(compund):
    """
    Used for convert iupac_name/smiles(to fix smiles) to canonical_smiles
    If not find return None else return String
    compund : String  convert canonical_smiles
    """
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(compund) + '/smiles'
        ans = urlopen(url).read().decode('utf8')
        return ans

    except:
        return None


def tran_iso2can_rdkit(smiles):
    """
    used for converting canonical_smiles to smiles through pubchem,speed : fast
    If not find return None else return String

    smiles : String
    """

    try:
        isomericsmiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True, isomericSmiles=False)

    except Exception:
        # Catch any other unexpected errors
        print(f"Unexpected error: \r\n{smiles}")
        return None

    if isomericsmiles:
        return isomericsmiles

    else:
        print(f'SMILES not find now ans is {smiles}\r\n'
              f'{isomericsmiles}')
        return isomericsmiles


def calculate_ecfp_rdkit(smiles, radius=2, n_bits=1024):
    """
    used for calculate_ecfp
    If not calculate not finish return None else return ecfp list

    smiles : String [smiles of the compound]
    radius: int  (default : 2)
    n_bits : int [size of calculation] (default :1024)

    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:  # 如果SMILES字符串有效
        ecfp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return list(ecfp)  # 转换为list数组
    else:
        return None  # 如果SMILES无效，则返回一个全0的指纹


def tran_iupac2smiles_fun(compound, smiles_type='CANONSMILES'):
def tran_iupac2smiles_fun(compound, smiles_type='CANONSMILES'):
    """
    input iupac/smiles trans it to isomericsmiles
    
    compound : String [smiles or iupacname of the compound]
    """
    smiles = tran_iupac2can_smiles_cir(compound) if compound else None
    smiles = tran_iupac2can_smiles_cir(compound) if compound else None

    print(f'{compound}\r\n'
          f'canonical_smiles: {smiles}\r\n')
          f'canonical_smiles: {smiles}\r\n')

    if smiles_type == 'CANONSMILES':
        smiles = tran_iso2can_rdkit(smiles) if smiles else None
    elif smiles_type == 'ISOMERICSMILES':
        smiles = restful_pub_finder(smiles, SMILES_BASE_FINDER)
    if smiles_type == 'CANONSMILES':
        smiles = tran_iso2can_rdkit(smiles) if smiles else None
    elif smiles_type == 'ISOMERICSMILES':
        smiles = restful_pub_finder(smiles, SMILES_BASE_FINDER)

    if smiles:
        print(f"transinto smiles with :{smiles}")
    if smiles:
        print(f"transinto smiles with :{smiles}")
    else:
        print("get_smiles failed")

    return smiles
    return smiles
