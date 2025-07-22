import pymysql
import pandas as pd
from sqlalchemy import create_engine

# 连接参数
config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'rootroot',
    'database': 'testpython',
    'port': 3306,
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor  # 返回字典格式结果
}

try:
    # 建立连接
    connection = pymysql.connect(**config)

    # 方法1：使用原生SQL查询
    with connection.cursor() as cursor:
        # 执行查询
        cursor.execute("SELECT version()")
        db_version = cursor.fetchone()
        print(f"MySQL版本: {db_version['version()']}")

        # 参数化查询（防止SQL注入）
        cursor.execute("SELECT * FROM products WHERE price > %s", (50.0,))
        products = cursor.fetchall()
        print("\n高价产品:")
        for product in products:
            print(f"{product['name']} - ${product['price']}")

    # 方法2：使用Pandas直接操作
    engine = create_engine(
        f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}/{config['database']}"
    )

    # 从数据库读取到DataFrame
    df = pd.read_sql("SELECT * FROM orders", engine)
    print("\n订单数据:")
    print(df.info())

    # 将DataFrame写入新表
    new_data = pd.DataFrame({
        'id': [101, 102],
        'product': ['Laptop', 'Phone'],
        'quantity': [2, 3]
    })
    new_data.to_sql('new_orders', engine, if_exists='append', index=False)
    print("\n数据写入完成")

except pymysql.Error as e:
    print(f"数据库错误: {e}")

finally:
    if 'connection' in locals():
        connection.close()
        print("数据库连接已关闭")