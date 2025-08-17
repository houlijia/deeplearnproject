import pandas as pd
from sqlalchemy import create_engine, text


class ExcelDBManager:
    def __init__(self, db_type='mysql', user='root', password='pass', host='localhost', port='3306', db_name='test_db'):
        """初始化数据库连接"""
        self.db_uri = f"{db_type}+pymysql://{user}:{password}@{host}:{port}/{db_name}"
        self.engine = create_engine(self.db_uri)

    def excel_to_db(self, excel_path, table_name, sheet_name=0):
        """Excel导入数据库"""
        try:
            # 读取Excel（支持xls/xlsx）
            df = pd.read_excel(excel_path, sheet_name=sheet_name)

            # 写入数据库（自动建表/追加）
            df.to_sql(
                name=table_name,
                con=self.engine,
                index=False,
                if_exists='replace'  # 'append'追加/'replace'覆盖
            )
            print(f"成功导入 {len(df)} 行数据到表 {table_name}")
            return True
        except Exception as e:
            print(f"导入失败: {str(e)}")
            return False

    def db_to_excel(self, table_name, excel_path, query=None):
        """数据库导出到Excel"""
        try:
            # 自定义查询或全表导出
            sql = query if query else f"SELECT * FROM {table_name}"

            # 执行查询并转为DataFrame
            with self.engine.connect() as conn:
                df = pd.read_sql(text(sql), conn)

            # 写入Excel（xlsx格式）
            df.to_excel(excel_path, index=False, engine='openpyxl')
            print(f"成功导出 {len(df)} 行数据到 {excel_path}")
            return True
        except Exception as e:
            print(f"导出失败: {str(e)}")
            return False


# ================== 使用示例 ==================
if __name__ == "__main__":
    # 1. 初始化连接（修改为你的数据库配置）
    manager = ExcelDBManager(
        user='root',
        password='rootroot',
        db_name='mysqlpython'
    )

    # 2. 从Excel导入数据库
    manager.excel_to_db(
        excel_path="./resource/testexcel.xlsx",  # 输入文件路径
        table_name="products",  # 数据库表名
        sheet_name="基本信息"  # Excel工作表名
    )

    # 3. 从数据库导出到Excel
    manager.db_to_excel(
        table_name="products",  # 数据库表名
        excel_path="./resource/output_data.xlsx",  # 输出文件路径
        # query="SELECT * FROM products WHERE price > 100"  # 可选自定义查询
    )