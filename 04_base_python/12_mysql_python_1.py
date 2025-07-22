import mysql.connector
from mysql.connector import Error

try:
    # 创建数据库连接
    connection = mysql.connector.connect(
        host='localhost',
        user='your_username',
        password='your_password',
        database='your_database',
        port=3306,
        # 解决公钥检索问题
        allow_local_infile=True,
        auth_plugin='mysql_native_password'
    )

    if connection.is_connected():
        print("成功连接到MySQL数据库")

        # 创建游标对象
        cursor = connection.cursor()

        # 示例1：执行查询
        cursor.execute("SELECT * FROM your_table LIMIT 5")
        results = cursor.fetchall()

        print("\n查询结果：")
        for row in results:
            print(row)

        # 示例2：插入数据
        insert_query = "INSERT INTO users (name, email) VALUES (%s, %s)"
        data = ('John Doe', 'john@example.com')
        cursor.execute(insert_query, data)
        connection.commit()
        print(f"\n插入成功，影响行数: {cursor.rowcount}")

        # 示例3：使用Pandas读取数据
        import pandas as pd

        df = pd.read_sql("SELECT * FROM your_table", connection)
        print("\nPandas DataFrame:")
        print(df.head())

except Error as e:
    print(f"数据库错误: {e}")

finally:
    # 关闭连接
    if 'connection' in locals() and connection.is_connected():
        cursor.close()
        connection.close()
        print("\nMySQL连接已关闭")