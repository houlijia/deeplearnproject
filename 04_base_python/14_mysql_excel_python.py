import pandas as pd

from sqlalchemy import create_engine, text
from mylogger.mylogger import *


class ExcelDbManager:
    def __init__(self, db_type='mysql', user='root', password='1234', host='localhost', port='3306', db_name='test_db'):
        self.db_url = f"{db_type}+pymysql://{user}:{password}@{host}:{port}/{db_name}"
        self.db_engine = create_engine(self.db_url)

        mylogger.info(
            f"db_type={db_type},"
            f"host={host},"
            f"port={port},"
            f"db_name={db_name}"
        )

    def excel_to_db(self, excel_path, table_name, sheet_name=0):
        try:
            # 读取excel
            excel_data = pd.read_excel(excel_path, sheet_name=sheet_name)

            # 写入数据库
            result = excel_data.to_sql(
                name=table_name,
                con=self.db_engine,
                index=False,
                if_exists='replace'  # 'append'追加 / 'replace' 覆盖
            )
            if result == len(excel_data):
                print(f"成功导入 {len(excel_data)} 行数据到表: {table_name}")
                return True
        except Exception as e:
            print(f"倒入失败：{str(e)}")
            return False

    def db_to_excel(self, table_name, excel_path, query=None):
        """数据库数据到Excel"""
        try:
            # sql查询全量数据
            sql = query if query else f"SELECT * FROM {table_name}"

            # 执行查询并转换为DataFrame
            with self.db_engine.connect() as con:
                excel_data = pd.read_sql(text(sql), con)

            #  将数据写入到execl里面
            excel_data.to_excel(excel_path, index=False, engine='openpyxl')
            print(f"成功导出 {len(excel_data)} 行数据到 {excel_path}")
            return True

        except Exception as e:
            print(f"导出失败: {str(e)}")
            return False


if __name__ == "__main__":
    dbmanager = ExcelDbManager(
        user='root',
        password='rootroot',
        db_name='mysqlpython'
    )

    dbmanager.excel_to_db(
        table_name="products",  # 数据库表名
        excel_path="./resource/testexcel.xlsx",  # 输入文件路径
        sheet_name="基本信息"  # Excel工作表名
    )

    dbmanager.db_to_excel(
        table_name="products",  # 数据库表名
        excel_path="./resource/output_data.xlsx",  # 输出文件路径
    )
