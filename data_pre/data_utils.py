import time
from urllib.request import urlopen
from urllib.parse import quote
import urllib.parse
import requests
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def tran_smiles_iso2can_pub(canonical_smiles):
    """
    used for converting canonical_smiles to smiles through pubchem,speed : slowly
    If not find return None else return String
    
    canonical_smiles : String
    """ 

    if canonical_smiles:
        # 将Smiles名称进行URL编码
        canonical_smiles_url = urllib.parse.quote(canonical_smiles,safe='')
        # 构建URL
        # url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{canonical_smiles_url}/property/IsomericSMILES/TXT"
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/property/IsomericSMILES/TXT?smiles={canonical_smiles_url}"
        if canonical_smiles_url:
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
                            print(f"No SMILES data found for {canonical_smiles}")
                            return None
                        
                        # 返回SMILES编码
                        return smiles_pbc
                    
                    # 如果返回的状态码不是200，打印错误并退出
                    else:
                        print(f"Error: {response.status_code} - Failed to retrieve data for {canonical_smiles}")
                        return None
                
                except requests.RequestException as e:
                    # 捕获网络请求异常，打印错误信息并等待2秒重试
                    print(f"Network error: {e}. Retrying in 2 seconds...")
                    time.sleep(1)
        else:
            return None

def tran_name2iso_pub(name):
    """
    used for converting canonical_smiles to smiles through pubchem,speed : slowly
    If not find return None else return String
    
    canonical_smiles : String
    """ 

    if name:
        # 将Smiles名称进行URL编码
        name_url = urllib.parse.quote(name,safe='')
        # 构建URL
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name_url}/property/IsomericSMILES/TXT"
        if name_url:
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
                            print(f"No SMILES data found for {name}")
                            return None
                        
                        # 返回SMILES编码
                        if '\n' in smiles_pbc:
                            smiles_pbc = None
                        print(f'return smiles {smiles_pbc}')
                        return smiles_pbc
                    
                    # 如果返回的状态码不是200，打印错误并退出
                    else:
                        print(f"Error: {response.status_code} - Failed to retrieve data for {name}")
                        return None
                
                except requests.RequestException as e:
                    # 捕获网络请求异常，打印错误信息并等待2秒重试
                    print(f"Network error: {e}. Retrying in 2 seconds...")
                    time.sleep(1)
        else:
            return None

def tran_iupac2smiles_cir(compund):
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
    isomericsmiles = None
    try:
        isomericsmiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True, isomericSmiles=False)
    except Exception:
        # Catch any other unexpected errors
        print(f"Unexpected error: \r\n{smiles}")
        return None
    
    if isomericsmiles:
        return isomericsmiles

    else:
        #print(f'SMILES not find now ans is {smiles}\r\n{isomericsmiles}')
        return None

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
        return list(ecfp) # 转换为list数组
    else:
        return None  # 如果SMILES无效，则返回一个全0的指纹

def tran_iupac2smiles_fun(compound):
    """
    input iupac/smiles trans it to isomericsmiles
    
    compound : String [smiles or iupacname of the compound]
    """
    canonical_smiles = tran_iupac2smiles_cir(compound) if compound else None

    isomericsmiles = tran_iso2can_rdkit(canonical_smiles) if canonical_smiles else None

    if isomericsmiles:
        print(f"transinto smiles with :{isomericsmiles}")
    else:
        print("get_smiles failed")

    return isomericsmiles
