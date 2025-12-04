import mysql.connector
import sys
import os

current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from app.config import DB_CONFIG


def init_database():
    # 1. 准备连接参数
    init_config = {
        'host': DB_CONFIG['host'],
        'user': DB_CONFIG['user'],
        'password': DB_CONFIG['password']
    }
    target_db_name = DB_CONFIG['database']

    try:
        print(">>> 正在连接 MySQL 服务器...")
        conn = mysql.connector.connect(**init_config)
        cursor = conn.cursor()

        # 2. 创建并选择数据库
        print(f">>> 检查/创建数据库: {target_db_name}")
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {target_db_name}")
        cursor.execute(f"USE {target_db_name}")

        # 3. 定义表结构
        tables = {
            "Flower": """
                CREATE TABLE Flower (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    category VARCHAR(50) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "Clothes": """
                CREATE TABLE Clothes (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    category VARCHAR(50) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "Titanic": """
                CREATE TABLE Titanic (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    pclass INT NOT NULL,
                    sex INT NOT NULL,
                    age FLOAT,
                    sibsp INT,
                    parch INT,
                    fare FLOAT,
                    embarked INT,
                    survival_probability FLOAT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "Nonlinear": """
                CREATE TABLE Nonlinear (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    feature1 FLOAT NOT NULL,
                    feature2 FLOAT NOT NULL,
                    feature3 FLOAT NOT NULL,
                    feature4 FLOAT NOT NULL,
                    feature5 FLOAT NOT NULL,
                    feature6 FLOAT NOT NULL,
                    feature7 FLOAT NOT NULL,
                    feature8 FLOAT NOT NULL,
                    prediction1 FLOAT NOT NULL,
                    prediction2 FLOAT NOT NULL,
                    prediction3 FLOAT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
        }

        # 4. 执行创建
        print("\n>>> 开始初始化表结构...")
        for table_name, create_sql in tables.items():
            # 先删后建，确保结构最新
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            cursor.execute(create_sql)
            print(f"    Table '{table_name}' ... [Created]")

        conn.commit()
        print(f"\n🎉 数据库 {target_db_name} 初始化成功！")

    except mysql.connector.Error as err:
        print(f"\n❌ 数据库错误: {err}")
    
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    init_database()