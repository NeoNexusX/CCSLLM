import time
from urllib.request import urlopen
from urllib.parse import quote
import urllib.parse
import requests
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"

# query_arg
NAME_BASE_FINDER = "/name/property"
SMILES_BASE_FINDER = "/smiles/property"

ISOMERIC_SMILES_RESULT = "/IsomericSMILES"
URL_TXT_END = "/TXT"
SMILES_QUERY_ARG = "?smiles="


def restful_pub_finder(query_arg, query_base=NAME_BASE_FINDER):
    """
    used for converting canonical_smiles to smiles through pubchem,speed : slowly
    If not find return None else return String
    
    canonical_smiles : String
    """

    if query_arg:
        # 将Smiles名称进行URL编码
        query_arg_url = urllib.parse.quote(query_arg, safe='')
        # 构建URL
        url = BASE_URL + query_base + ISOMERIC_SMILES_RESULT + URL_TXT_END + SMILES_QUERY_ARG + f"{query_arg_url}"
        if query_arg_url:
            while True:
                try:
                    # 发起GET请求
                    response = requests.get(url)

                    # 如果请求成功且返回200
                    if response.status_code == 200:
                        # 处理响应内容，提取SMILES编码
                        smiles_pbc = response.text.strip()

                        # 如果返回数据为空，表示没有对应数据
                        if not smiles_pbc:
                            print(f"No SMILES data found for {query_arg}")
                            return None

                        # 返回SMILES编码
                        if '\n' in smiles_pbc:
                            smiles_pbc = None
                        print(f"{query_arg}\r\nsmiles_pbc find :{smiles_pbc}\r\n")
                        return smiles_pbc

                    # 如果返回的状态码不是200，打印错误并退出
                    else:
                        print(f"Error: {response.status_code} - Failed to retrieve data for {query_arg}")
                        return None

                except requests.RequestException as e:
                    # 捕获网络请求异常，打印错误信息并等待2秒重试
                    print(f"Network error: {e}. Retrying in 2 seconds...")
                    time.sleep(1)
        else:
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


def tran_iupac2smiles_fun(compound):
    """
    input iupac/smiles trans it to isomericsmiles
    
    compound : String [smiles or iupacname of the compound]
    """
    canonical_smiles = tran_iupac2can_smiles_cir(compound) if compound else None

    print(f'{compound}\r\n'
          f'canonical_smiles: {canonical_smiles}\r\n')

    isomericsmiles = tran_iso2can_rdkit(canonical_smiles) if canonical_smiles else None

    if isomericsmiles:
        print(f"transinto smiles with :{isomericsmiles}")
    else:
        print("get_smiles failed")

    return isomericsmiles
