import os
import re
import time
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd

# 获取脚本所在目录（确保所有文件保存在此目录）
script_dir = os.path.dirname(os.path.abspath(__file__))

# 配置日志（保存到脚本目录）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=os.path.join(script_dir, '行业板块爬取日志.log')
)
logger = logging.getLogger(__name__)


class SinaIndustryScraper:
    def __init__(self):
        # 浏览器配置
        self.options = webdriver.ChromeOptions()
        self.options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
        self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=self.options
        )
        # 规避反爬检测
        self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        })
        self.target_url = 'https://finance.sina.com.cn/stock/sl/#industry_1'
        self.data = []  # 存储提取的11个字段数据

    def get_script_dir_file(self, filename):
        """生成脚本目录下的文件路径"""
        return os.path.join(script_dir, filename)

    def fetch_page(self):
        """获取页面并保存源码到脚本目录"""
        try:
            self.driver.get(self.target_url)
            # 等待表格加载（最长30秒）
            WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.TAG_NAME, 'table'))
            )
            # 确保数据完全渲染
            time.sleep(2)
            # 保存页面源码到脚本目录（用于调试）
            page_path = self.get_script_dir_file('页面源码.html')
            with open(page_path, 'w', encoding='utf-8') as f:
                f.write(self.driver.page_source)
            logger.info(f"页面源码已保存至：{page_path}")
            return self.driver.page_source
        except Exception as e:
            logger.error(f"页面加载失败：{str(e)}")
            return None
        finally:
            self.driver.quit()

    def clean_number(self, text, is_float=False):
        """清洗数字（去除逗号、空格，转换类型）"""
        if not text.strip():
            return None
        cleaned = re.sub(r'[,\s]', '', text.strip())
        try:
            return float(cleaned) if is_float else int(cleaned)
        except:
            return text  # 保留原始文本（用于调试）

    def clean_percent(self, text):
        """清洗涨跌幅（去除%，转换为浮点数）"""
        if not text.strip():
            return None
        cleaned = re.sub(r'[%\s]', '', text.strip())
        try:
            return float(cleaned)
        except:
            return text  # 保留原始文本

    def parse_table(self, html):
        """精准提取11个指定字段"""
        if not html:
            return

        soup = BeautifulSoup(html, 'lxml')
        # 定位数据表格（优先选择行数量最多的表格）
        tables = soup.find_all('table')
        if not tables:
            logger.error("未找到表格")
            return
        target_table = max(tables, key=lambda t: len(t.find_all('tr')))  # 选行数最多的表格
        rows = target_table.find_all('tr')
        if len(rows) < 2:
            logger.error("表格无数据行")
            return

        # 解析表头（用于校验字段顺序）
        header_cells = rows[0].find_all(['th', 'td'])
        headers = [cell.text.strip() for cell in header_cells]
        logger.info(f"表头字段：{headers}")  # 日志打印表头，方便核对

        # 解析数据行（从第二行开始）
        for row in rows[1:]:
            cells = row.find_all('td')
            if len(cells) < 11:  # 确保至少有11列（匹配11个字段）
                continue

            try:
                # 严格对应11个字段（按页面顺序）
                row_data = {
                    # 行业基础数据
                    '板块': cells[0].text.strip(),
                    '公司家数': self.clean_number(cells[1].text.strip()),
                    '平均价格': self.clean_number(cells[2].text.strip(), is_float=True),
                    '涨跌额': self.clean_number(cells[3].text.strip(), is_float=True),
                    '涨跌幅(%)': self.clean_percent(cells[4].text.strip()),
                    '总成交量(手)': self.clean_number(cells[5].text.strip()),
                    '总成交额(万元)': self.clean_number(cells[6].text.strip(), is_float=True),
                    # 领涨股数据（区分行业字段，避免重名）
                    '领涨股': cells[7].text.strip(),
                    '领涨股涨跌幅(%)': self.clean_percent(cells[8].text.strip()),
                    '领涨股当前价': self.clean_number(cells[9].text.strip(), is_float=True),
                    '领涨股涨跌额': self.clean_number(cells[10].text.strip(), is_float=True)
                }
                self.data.append(row_data)
                logger.debug(f"已提取：{row_data['板块']}")
            except Exception as e:
                logger.warning(f"行解析失败：{str(e)}，行内容：{row.text[:100]}")

    def save_data(self):
        """保存CSV到脚本目录"""
        if not self.data:
            print("未提取到有效数据，请查看日志和【页面源码.html】")
            return
        # CSV文件保存到脚本目录
        csv_path = self.get_script_dir_file('行业板块数据.csv')
        df = pd.DataFrame(self.data)
        # 确保字段顺序与需求一致
        df = df[[
            '板块', '公司家数', '平均价格', '涨跌额', '涨跌幅(%)',
            '总成交量(手)', '总成交额(万元)', '领涨股',
            '领涨股涨跌幅(%)', '领涨股当前价', '领涨股涨跌额'
        ]]
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"数据提取完成！共 {len(self.data)} 条记录")
        print(f"CSV文件已保存至：{csv_path}")
        logger.info(f"数据已保存至：{csv_path}")

    def run(self):
        print("开始爬取行业板块数据...")
        html = self.fetch_page()
        if not html:
            print("爬取失败，请查看日志：行业板块爬取日志.log")
            return
        self.parse_table(html)
        self.save_data()


if __name__ == "__main__":
    scraper = SinaIndustryScraper()
    scraper.run()