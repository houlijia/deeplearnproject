import mysql.connector
from mysql.connector import Error

try:
    # 创建数据库连接
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='rootroot',
        database='testpython',
        port='3306'
    )

    if connection.is_connected():
        print("成功连接至MySQL数据库")
        a = locals()
        # 创建游标对象
        cursor = connection.cursor()

        # 查询语句
        cursor.execute("SELECT * FROM employees LIMIT 5")
        results = cursor.fetchall()
        print("\n 查询结果：")

        for row in results:
            print(row)

        # 示例2: 插入数据
        insert_sql = "INSERT INTO employees (name,email) VALUES (%s, %s)"
        data = ('lisi', '1234@163.com')
        cursor.execute(insert_sql, data)
        connection.commit()
        print(f'\n 插入成功，影响行数：{cursor.rowcount}')

finally:
    # 检查变量名 'connection' 是否存在于当前作用域的局部变量中
    # locals() 返回类似这样的字典：{'connection': <mysql.connector对象>, 'cursor': <游标对象>, ...}
    # 这可以防止当 connection 变量不存在时直接访问 connection.is_connected() 导致 NameError
    # 为什么需要这样写？
    # 在数据库操作的异常处理中：
    # 如果数据库连接建立失败（比如认证错误），connection 变量可能根本不存在
    # 如果连接已经关闭，尝试再次关闭会导致错误
    # 使用 locals() 检查可以安全地处理这些边界情况

    if 'connection' in locals() and connection.is_connected():
    # 写法2 :
    # # 使用变量检查
    # if 'connection' in globals() and connection.is_connected():
        cursor.close()
        connection.close()
        print("\n MySQL连接已关闭")

    # 写法3 :
    # # 更安全的写法（避免使用 locals()）
    # try:
    #     if connection and connection.is_connected():
    #         cursor.close()
    #         connection.close()
    #         print("\nMySQL连接已关闭")
    # except NameError:  # 处理 connection 未定义的情况
    #     pass
