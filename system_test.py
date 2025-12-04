import os
import sys

# ==========================================
# 0. 全局配置 (最优先执行)
# ==========================================
# 1. 屏蔽 TensorFlow/Metal 冗余日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# 2. 动态路径配置
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
if os.path.basename(current_dir) == 'app':
    project_root = os.path.dirname(current_dir)
else:
    project_root = current_dir
if project_root not in sys.path:
    sys.path.append(project_root)

# 3. 导入依赖
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import joblib
import pickle
import mysql.connector
tf.get_logger().setLevel('ERROR') 

from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

try:
    from app.config import MODEL_DIRS, DATA_DIRS, DB_CONFIG
except ImportError:
    print("❌ 配置导入失败，请检查 app/config.py")
    sys.exit(1)

# 控制台颜色
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# PyTorch 类定义
class DeepNeuralNet(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=8, output_dim=1, epochs=200, lr=0.002):
        self.model = None
        self.device = torch.device('cpu')
    def predict(self, X):
        if self.model is None: return np.zeros(len(X))
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.model(X_t).cpu().numpy().ravel() if self.output_dim == 1 else self.model(X_t).cpu().numpy()

# 数据库验证工具
def verify_db_record(table_name, check_col, check_val, tolerance=None):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT 1")
        record = cursor.fetchone()
        conn.close()
        
        if not record: return False, "无记录", None
        
        # 映射常用列索引
        col_idx = 1 # 默认 filename/feature1
        if table_name == 'Titanic' and check_col == 'fare': col_idx = 6
        
        db_val = record[col_idx]
        
        if tolerance:
            match = abs(float(db_val) - float(check_val)) < tolerance
        else:
            match = str(db_val) == str(check_val)
            
        return match, db_val, record[0] 
    except Exception as e:
        return False, str(e), None

# ==========================================
# 1. 场景测试逻辑
# ==========================================

def run_scene1_test():
    print(f"\n{Colors.HEADER}>>> [场景 1] 花卉图像分类{Colors.ENDC}")
    results = {'perf': False, 'func': False}
    
    # --- A. 性能测试 ---
    print(f"{Colors.OKBLUE}[Performance]{Colors.ENDC} 评估测试集精度...")
    try:
        data_dir = tf.keras.utils.get_file('flower_photos', origin="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz", untar=True)
        val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2).flow_from_directory(
            data_dir, target_size=(150, 150), batch_size=32, class_mode='categorical', subset='validation', shuffle=False
        )
        model = load_model(os.path.join(MODEL_DIRS['scene1'], 'flowers.h5'))
        loss, acc = model.evaluate(val_gen, verbose=0)
        
        print(f"  - 样本数量: {val_gen.samples}")
        print(f"  - 识别精度: {acc:.2%}")
        if acc >= 0.85:
            print(f"  - 结论: {Colors.OKGREEN}PASSED (目标 >= 85%){Colors.ENDC}")
            results['perf'] = True
        else:
            print(f"  - 结论: {Colors.FAIL}FAILED{Colors.ENDC}")
    except Exception as e:
        print(f"  - 错误: {e}")

    # --- B. 功能测试 (含耗时) ---
    print(f"{Colors.OKBLUE}[Functional]{Colors.ENDC} 全链路业务模拟...")
    try:
        test_file = "test_auto_flower.jpg"
        dummy_img = np.random.random((1, 150, 150, 3)).astype(np.float32)
        print(f"  1. 接收输入: 虚拟图像 (150x150x3)")
        
        model_func = load_model(os.path.join(MODEL_DIRS['scene1'], 'flowers.h5'))
        
        # 计时开始
        t_start = time.time()
        pred = model_func.predict(dummy_img, verbose=0)
        t_cost = (time.time() - t_start) * 1000 # 转换为毫秒
        
        pred_idx = np.argmax(pred)
        print(f"  2. 模型推理: 完成 (耗时={t_cost:.2f}ms, 预测索引={pred_idx})")
        
        conn = mysql.connector.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("INSERT INTO Flower (filename, category) VALUES (%s, %s)", (test_file, "Test_Daisy"))
        conn.commit()
        conn.close()
        print(f"  3. 写入数据库: 表[Flower]")
        
        success, val, db_id = verify_db_record("Flower", "filename", test_file)
        if success:
            print(f"  4. 数据验证: {Colors.OKGREEN}成功 (DB_ID={db_id}){Colors.ENDC}")
            results['func'] = True
        else:
            print(f"  4. 数据验证: {Colors.FAIL}失败{Colors.ENDC}")
    except Exception as e:
        print(f"  - 流程异常: {e}")
        
    return results

def run_scene2_test():
    print(f"\n{Colors.HEADER}>>> [场景 2] Titanic 生存预测{Colors.ENDC}")
    results = {'perf': False, 'func': False}
    
    # --- A. 性能测试 ---
    print(f"{Colors.OKBLUE}[Performance]{Colors.ENDC} 评估测试集精度...")
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values
        y = df['Survived'].values
        
        scaler = pickle.load(open(os.path.join(MODEL_DIRS['scene2'], 'scaler.pkl'), 'rb'))
        model = load_model(os.path.join(MODEL_DIRS['scene2'], 'taitanic.h5'))
        
        y_pred = (model.predict(scaler.transform(X), verbose=0) > 0.5).astype(int)
        acc = accuracy_score(y, y_pred)
        
        print(f"  - 预测精度: {acc:.2%}")
        if acc >= 0.70:
            print(f"  - 结论: {Colors.OKGREEN}PASSED (目标 >= 70%){Colors.ENDC}")
            results['perf'] = True
        else:
            print(f"  - 结论: {Colors.FAIL}FAILED{Colors.ENDC}")
    except Exception as e:
        print(f"  - 错误: {e}")

    # --- B. 功能测试 (含耗时) ---
    print(f"{Colors.OKBLUE}[Functional]{Colors.ENDC} 全链路业务模拟...")
    try:
        test_fare = 999.99
        # 构造单条数据
        raw_input = [3, 0, 22.0, 1, 0, test_fare, 0]
        input_scaled = scaler.transform([raw_input])
        print(f"  1. 接收输入: Pclass=3, Age=22, Fare={test_fare}...")
        
        # 计时
        t_start = time.time()
        prob = model.predict(input_scaled, verbose=0)[0][0]
        t_cost = (time.time() - t_start) * 1000
        
        print(f"  2. 模型推理: 完成 (耗时={t_cost:.2f}ms, 生存率={prob:.4f})")
        
        conn = mysql.connector.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("INSERT INTO Titanic (pclass, sex, age, sibsp, parch, fare, embarked, survival_probability) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                    (3, 0, 22, 1, 0, test_fare, 0, float(prob)))
        conn.commit()
        conn.close()
        print(f"  3. 写入数据库: 表[Titanic]")
        
        success, val, db_id = verify_db_record("Titanic", "fare", test_fare, tolerance=0.01)
        if success:
            print(f"  4. 数据验证: {Colors.OKGREEN}成功 (DB_ID={db_id}){Colors.ENDC}")
            results['func'] = True
        else:
            print(f"  4. 数据验证: {Colors.FAIL}失败{Colors.ENDC}")
    except Exception as e:
        print(f"  - 流程异常: {e}")
        
    return results

def run_scene3_test():
    print(f"\n{Colors.HEADER}>>> [场景 3] 时尚服饰分类{Colors.ENDC}")
    results = {'perf': False, 'func': False}
    
    # --- A. 性能测试 ---
    print(f"{Colors.OKBLUE}[Performance]{Colors.ENDC} 评估测试集精度...")
    try:
        (_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_test_norm = x_test.reshape(-1, 28, 28, 1) / 255.0
        model = load_model(os.path.join(MODEL_DIRS['scene3'], 'my_model.h5'))
        loss, acc = model.evaluate(x_test_norm, y_test, verbose=0)
        
        print(f"  - 识别精度: {acc:.2%}")
        if acc >= 0.80:
            print(f"  - 结论: {Colors.OKGREEN}PASSED (目标 >= 80%){Colors.ENDC}")
            results['perf'] = True
        else:
            print(f"  - 结论: {Colors.FAIL}FAILED{Colors.ENDC}")
    except Exception as e:
        print(f"  - 错误: {e}")

    # --- B. 功能测试 (含耗时) ---
    print(f"{Colors.OKBLUE}[Functional]{Colors.ENDC} 全链路业务模拟...")
    try:
        test_file = "test_auto_coat.png"
        dummy_img = np.random.random((1, 28, 28, 1)).astype(np.float32)
        print(f"  1. 接收输入: 灰度图像 (28x28x1)")
        
        model_func = load_model(os.path.join(MODEL_DIRS['scene3'], 'my_model.h5'))
        
        # 计时
        t_start = time.time()
        res = model_func.predict(dummy_img, verbose=0)
        t_cost = (time.time() - t_start) * 1000
        
        pred_cls = np.argmax(res)
        print(f"  2. 模型推理: 完成 (耗时={t_cost:.2f}ms, 类别={pred_cls})")
        
        conn = mysql.connector.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("INSERT INTO Clothes (filename, category) VALUES (%s, %s)", (test_file, str(pred_cls)))
        conn.commit()
        conn.close()
        print(f"  3. 写入数据库: 表[Clothes]")
        
        success, val, db_id = verify_db_record("Clothes", "filename", test_file)
        if success:
            print(f"  4. 数据验证: {Colors.OKGREEN}成功 (DB_ID={db_id}){Colors.ENDC}")
            results['func'] = True
        else:
            print(f"  4. 数据验证: {Colors.FAIL}失败{Colors.ENDC}")
    except Exception as e:
        print(f"  - 流程异常: {e}")
    return results

def run_scene4_test():
    print(f"\n{Colors.HEADER}>>> [场景 4] 非线性系统回归{Colors.ENDC}")
    results = {'perf': False, 'func': False}
    
    # --- A. 性能测试 ---
    print(f"{Colors.OKBLUE}[Performance]{Colors.ENDC} 复现数据流并评估...")
    try:
        data_root = DATA_DIRS['scene4']
        x_tr = pd.read_csv(os.path.join(data_root, 'traindata/x_train'), sep='\s+', header=None).values
        y_tr = pd.read_csv(os.path.join(data_root, 'traindata/y_train'), sep='\s+', header=None).values
        x_te = pd.read_csv(os.path.join(data_root, 'testdata/x_test'), sep='\s+', header=None).values
        y_te = pd.read_csv(os.path.join(data_root, 'testdata/y_test'), sep='\s+', header=None).values
        X_all = np.vstack([x_tr, x_te])
        y_all = np.vstack([y_tr, y_te])
        keep_mask = np.ones(len(X_all), dtype=bool)
        for i in range(3):
            est = HistGradientBoostingRegressor(max_iter=50, random_state=42).fit(X_all, y_all[:, i])
            errors = np.abs(est.predict(X_all) - y_all[:, i])
            keep_mask &= (errors <= np.quantile(errors, 0.80))
        X_clean = X_all[keep_mask]
        
        base_path = MODEL_DIRS['scene4']
        poly = joblib.load(os.path.join(base_path, 'poly.pkl'))
        scaler = joblib.load(os.path.join(base_path, 'scaler_x.pkl'))
        models = joblib.load(os.path.join(base_path, 'models.pkl'))
        X_poly = poly.transform(X_clean)
        _, X_val_raw, _, y_val = train_test_split(X_poly, y_all[keep_mask], test_size=0.15, random_state=42)
        X_val_s = scaler.transform(X_val_raw)
        
        r2_scores = []
        for i in range(3):
            y_pred = models[i].predict(X_val_s)
            r2_scores.append(r2_score(y_val[:, i], y_pred))
        avg_r2 = np.mean(r2_scores)
        
        print(f"  - 验证集 R2: {avg_r2:.4f}")
        if avg_r2 >= 0.70:
            print(f"  - 结论: {Colors.OKGREEN}PASSED (目标 >= 0.70){Colors.ENDC}")
            results['perf'] = True
        else:
            print(f"  - 结论: {Colors.FAIL}FAILED{Colors.ENDC}")
    except Exception as e:
        print(f"  - 错误: {e}")

    # --- B. 功能测试 (含耗时) ---
    print(f"{Colors.OKBLUE}[Functional]{Colors.ENDC} 全链路业务模拟...")
    try:
        marker = 777.777
        raw_input = [0.5] * 8
        print(f"  1. 接收输入: 特征向量 [0.5, ...]")
        
        # 预处理
        feat_poly = poly.transform([raw_input])
        feat_scaled = scaler.transform(feat_poly)
        
        # 计时 (Stacking 3个维度)
        t_start = time.time()
        preds = [m.predict(feat_scaled)[0] for m in models]
        t_cost = (time.time() - t_start) * 1000
        
        print(f"  2. 模型推理: 完成 (耗时={t_cost:.2f}ms, 输出={np.round(preds, 2)})")
        
        conn = mysql.connector.connect(**DB_CONFIG)
        cur = conn.cursor()
        sql = "INSERT INTO Nonlinear (feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,prediction1,prediction2,prediction3) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        cur.execute(sql, (marker, *raw_input[1:], *preds))
        conn.commit()
        conn.close()
        print(f"  3. 写入数据库: 表[Nonlinear]")
        
        success, val, db_id = verify_db_record("Nonlinear", "feature1", marker, tolerance=0.001)
        if success:
            print(f"  4. 数据验证: {Colors.OKGREEN}成功 (DB_ID={db_id}){Colors.ENDC}")
            results['func'] = True
        else:
            print(f"  4. 数据验证: {Colors.FAIL}失败{Colors.ENDC}")
    except Exception as e:
        print(f"  - 流程异常: {e}")
        
    return results

if __name__ == "__main__":
    print(f"{Colors.BOLD}{'='*60}")
    print(f"   深度学习系统综合自动化测试 (System Test Suite)")
    print(f"{'='*60}{Colors.ENDC}")
    
    summary = {}
    summary['S1'] = run_scene1_test()
    summary['S2'] = run_scene2_test()
    summary['S3'] = run_scene3_test()
    summary['S4'] = run_scene4_test()
    
    print(f"\n{Colors.BOLD}{'='*60}")
    print(f"   最终测试报告 (Final Report)")
    print(f"{'='*60}{Colors.ENDC}")
    print(f"{'场景':<15} | {'性能测试':<15} | {'功能测试':<15}")
    print("-" * 50)
    
    all_pass = True
    for s, res in summary.items():
        p_status = "✅ PASS" if res['perf'] else "❌ FAIL"
        f_status = "✅ PASS" if res['func'] else "❌ FAIL"
        if not (res['perf'] and res['func']): all_pass = False
        print(f"{s:<15} | {p_status:<15} | {f_status:<15}")
        
    print("-" * 50)
    if all_pass:
        print(f"{Colors.OKGREEN}🏆 结论: 系统完全符合验收标准，准予发布！{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}⚠️ 结论: 存在未达标项，请检查日志修复。{Colors.ENDC}")