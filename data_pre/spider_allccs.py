import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import threading
from queue import Queue

# 读取CSV文件，获取目标ID列表
df_allccs = pd.read_csv("D:/Neo/allccs/allccs.csv")

# 假设CSV中包含名为 "AllCCS ID" 的列，我们读取该列并去重
target_ids = df_allccs["AllCCS ID"].dropna().unique().tolist()


# 设置全局 DataFrame 用于存储结果
columns = ["Adduct", "m/z", "CCS(Å2)", "Charge", "Instrument", "Type", "Approach", "Calibrant", "DOI", "Update date", "ID"]
df = pd.DataFrame(columns=columns)

# 登录函数
def login(driver):
    driver.get("http://allccs.zhulab.cn/login")
    
    # 填写用户名和密码
    driver.find_element(By.CSS_SELECTOR, 'input[placeholder="Username"]').send_keys("NeoNexus")
    driver.find_element(By.CSS_SELECTOR, 'input[placeholder="Password"]').send_keys("zhuzeyu6678")
    
    # 等待登录按钮加载并点击提交按钮
    login_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.btn.btn-primary.btn-block.btn-flat'))
    )
    login_button.click()
    
    # 等待登录后的页面加载，检查是否显示用户信息
    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "a.dropdown-toggle"))
        )
        print("Login successful!")
    except Exception as e:
        print(f"Login failed: {e}")
        driver.quit()
        exit()

# 获取每个ID的数据
def fetch_data_for_id(target_ids, driver, result_queue):
    # 登录一次，只在第一次登录时执行
    login(driver)
    
    new_data = []

    for target_id in target_ids:
        # 构造目标页面URL
        target_url = f"http://allccs.zhulab.cn/database/detail?ID={target_id}"

        # 打开目标页面
        driver.get(target_url)

        # 等待页面加载并确保 "Experimental CCS records" 区域可见
        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.PARTIAL_LINK_TEXT, "Experimental CCS records"))
            )
            print(f"Page with ID {target_id} loaded successfully!")

            # 获取页面内容并解析
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            # 查找所有包含 "Experimental CCS records" 的面板
            panels = soup.find_all('div', class_='panel box box-primary')
            for panel in panels:
                # 找到面板中的 "Experimental CCS records" 链接
                link = panel.find('a', string=lambda text: text and "Experimental CCS records" in text.strip())
                if link:
                    print(f"Found 'Experimental CCS records' in panel for ID {target_id}")

                    # 找到该面板中的表格
                    table = panel.find('table', class_='table table-striped table-hover')
                    if table:
                        rows = table.find_all('tr')
                        for row in rows[1:]:  # 排除表头
                            columns = row.find_all('td')
                            if len(columns) > 0:
                                # 提取数据
                                adduct = columns[0].text.strip()
                                mz = columns[1].text.strip()
                                ccs = columns[2].text.strip()
                                charge = columns[3].text.strip()
                                instrument = columns[4].text.strip()
                                ion_type = columns[5].text.strip()
                                approach = columns[6].text.strip()
                                calibrant = columns[7].text.strip()
                                
                                # 提取 DOI，处理 <a> 标签
                                doi_tag = columns[8].find('a')  # 查找 DOI 链接
                                doi = doi_tag['href'] if doi_tag else columns[8].text.strip()
                                
                                update_date = columns[9].text.strip()

                                # 打印每一行的提取数据（用于调试）
                                print(f"Extracted data for ID {target_id}: {adduct}, {mz}, {ccs}, {charge}, {instrument}, {ion_type}, {approach}, {calibrant}, {doi}, {update_date}")

                                # 将数据添加到当前页面的列表中
                                new_data.append({
                                    "Adduct": adduct,
                                    "m/z": mz,
                                    "CCS(Å2)": ccs,
                                    "Charge": charge,
                                    "Instrument": instrument,
                                    "Type": ion_type,
                                    "Approach": approach,
                                    "Calibrant": calibrant,
                                    "DOI": doi,
                                    "Update date": update_date,
                                    "ID": target_id
                                })
        except Exception as e:
            print(f"Error loading page for ID {target_id}: {str(e)}")
    
    # 将结果放入队列
    result_queue.put(new_data)

# 创建3个浏览器实例并在3个独立的线程中运行
def run_in_parallel():
    global df

    # 每个线程分配一部分目标ID进行抓取
    chunk_size = len(target_ids) // 3
    target_chunks = [
        target_ids[:chunk_size],
        target_ids[chunk_size:2*chunk_size],
        target_ids[2*chunk_size:]
    ]
    
    threads = []
    result_queue = Queue()

    # 创建3个浏览器实例并为每个实例启动一个线程
    for i in range(3):
        driver = webdriver.Chrome()  # 为每个线程创建一个独立的 WebDriver 实例
        thread = threading.Thread(target=fetch_data_for_id, args=(target_chunks[i], driver, result_queue), name=f"Thread-{i+1}")
        threads.append(thread)
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    # 从队列中获取每个线程的结果并合并到最终的 DataFrame 中
    all_results = []
    while not result_queue.empty():
        all_results.extend(result_queue.get())

    df = pd.DataFrame(all_results, columns=columns)
    df.to_csv("experimental_ccs_data.csv", index=False)
    print(f"Data saved to 'experimental_ccs_data.csv'")

if __name__ == "__main__":
    run_in_parallel()
