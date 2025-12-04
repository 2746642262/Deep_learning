import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 模型保存路径
MODEL_DIRS = {
    'scene1': os.path.join(BASE_DIR, 'scene1/models/'),
    'scene2': os.path.join(BASE_DIR, 'scene2/models/'),
    'scene3': os.path.join(BASE_DIR, 'scene3/models/'),
    'scene4': os.path.join(BASE_DIR, 'scene4/models/'),
}

# 2. 数据源路径
DATA_DIRS = {
    'scene4': os.path.join(BASE_DIR, 'scene4/data/')
}

# 确保模型目录存在
for path in MODEL_DIRS.values():
    os.makedirs(path, exist_ok=True)

# 3. 数据库配置
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "020929Lkx",
    "database": "Deep_learning"
}

