-- 创建 学生表(基础信息)
CREATE TABLE IF NOT EXISTS students(
id INT PRIMARY KEY AUTO_INCREMENT,
name VARCHAR(50) NOT NULL,
gender ENUM('男','女') DEFAULT '男',
birthdate DATE,
enroll_time DATETIME DEFAULT CURRENT_TIMESTAMP,
class_id INT NOT NULL,
score DECIMAL(5,2) CHECK(score BETWEEN 0 AND 100)
);

-- 订单表（关联操作）
CREATE TABLE IF NOT EXISTS orders(
	order_id INT PRIMARY KEY AUTO_INCREMENT,
	student_id INT,
	product VARCHAR(50),
	amount DECIMAL(10,2),
	order_date DATE,
	FOREIGN KEY (student_id) REFERENCES students(id)
);

-- 插入数据
INSERT INTO students (name,gender,birthdate, class_id, score) VALUES
('张三', '男', '2005-03-12', 1, 88.5),
('李四', '女', '2004-11-05', 1, 92.0),
('王五', '男', '2005-07-19', 2, 76.5),
('赵六', '女', '2003-09-23', 2, 95.5),
('钱七', '男', '2006-01-30', 3, 81.0);


-- 插入订单数据（关联查询测试）
INSERT INTO orders (student_id, product, amount, order_date) VALUES
(1, '数学教材', 120.50, '2023-09-01'),
(1, '文具套装', 45.80, '2023-10-15'),
(2, '英语词典', 98.00, '2023-08-22'),
(3, '实验器材', 220.00, '2023-11-03'),
(4, '编程书籍', 78.40, '2023-10-28'),
(5, '运动装备', 156.75, '2023-09-17');

-- 日期练习
-- 计算年龄（精确到年）
SELECT name,TIMESTAMPDIFF(YEAR,birthdate,CURDATE()) as age 
From students;

-- 本月订单
SELECT COUNT(*) AS order_count,SUM(amount) as total_amount
from orders 
WHERE MONTH(order_date) = MONTH(CURDATE());


-- 条件组合查询
SELECT * FROM students 
where class_id = 1
AND (score > 90 OR gender = '女');

-- LIKE 模糊查询
SELECT * FROM orders
WHERE product LIKE '%书%'
	AND amount BETWEEN 50 AND 100;


-- 聚合与分组统计
SELECT class_id, AVG(score) as avg_score
from students
GROUP BY class_id
HAVING avg_score > 85;

-- 学生订单总额TOP3
SELECT s.name, SUM(o.amount) as total_amount 
FROM students s JOIN orders o ON s.id = o.student_id
GROUP BY s.id
ORDER BY total_amount DESC
LIMIT 3;

-- 查看表占用的空间
SELECT 
table_name AS `表名`,
	ROUND((data_length + index_length) / (1024 * 1204), 2) as `大小(MB)`
from information_schema.TABLES
where table_schema = DATABASE();

  
  









