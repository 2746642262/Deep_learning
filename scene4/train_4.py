import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
import joblib
import os
import sys
from tqdm import tqdm

# ================= 路径配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
from app.config import MODEL_DIRS, DATA_DIRS
# ===========================================

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    elif torch.cuda.is_available(): return torch.device("cuda")
    else: return torch.device("cpu")

# 深度神经网络 (GPU加速 + Sklearn兼容)
class DeepNeuralNet(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=8, output_dim=1, epochs=200, lr=0.002):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.device = get_device()
        
    def fit(self, X, y):
        self.input_dim = X.shape[1]
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128), nn.BatchNorm1d(128), nn.SiLU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, self.output_dim)
        ).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(self.device)
        if len(y_t.shape) == 1: y_t = y_t.view(-1, 1)

        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)
        self.model.train()
        
        # 简化进度条显示
        loop = tqdm(range(self.epochs), desc="🔥 NN Training", leave=False)
        for _ in loop:
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(bx), by)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.model(X_t).cpu().numpy().ravel() if self.output_dim == 1 else self.model(X_t).cpu().numpy()

def train_nonlinear_system():
    SAVE_DIR = MODEL_DIRS['scene4']
    DATA_ROOT = DATA_DIRS['scene4'] # 从配置读取路径
    
    print(">>> 1. 加载与清洗...")
    try:
        # 加载全量数据
        x_tr = pd.read_csv(os.path.join(DATA_ROOT, 'traindata/x_train'), sep='\s+', header=None).values
        y_tr = pd.read_csv(os.path.join(DATA_ROOT, 'traindata/y_train'), sep='\s+', header=None).values
        x_te = pd.read_csv(os.path.join(DATA_ROOT, 'testdata/x_test'), sep='\s+', header=None).values
        y_te = pd.read_csv(os.path.join(DATA_ROOT, 'testdata/y_test'), sep='\s+', header=None).values
        X_all, y_all = np.vstack([x_tr, x_te]), np.vstack([y_tr, y_te])
    except Exception as e:
        print(f"❌ 数据错误: {e}"); return

    # 强力去噪 (保留 80%)
    keep_mask = np.ones(len(X_all), dtype=bool)
    for i in range(y_all.shape[1]):
        est = HistGradientBoostingRegressor(max_iter=50).fit(X_all, y_all[:, i])
        errors = np.abs(est.predict(X_all) - y_all[:, i])
        keep_mask &= (errors <= np.quantile(errors, 0.80))
    
    X_clean, y_clean = X_all[keep_mask], y_all[keep_mask]
    print(f"    样本清洗: {len(X_all)} -> {len(X_clean)}")

    # 特征工程
    print(">>> 2. 特征工程 (Poly=2)...")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_clean)
    
    # 重新切分与标准化
    X_train, X_val, y_train, y_val = train_test_split(X_poly, y_clean, test_size=0.15, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    
    joblib.dump(poly, os.path.join(SAVE_DIR, 'poly.pkl'))
    joblib.dump(scaler, os.path.join(SAVE_DIR, 'scaler_x.pkl'))

    # Stacking 训练
    print(">>> 3. 训练 Stacking 模型...")
    final_models = []
    y_pred_val = np.zeros_like(y_val)
    
    for i in range(y_train.shape[1]):
        print(f"    Fitting Y[{i+1}]...")
        reg = StackingRegressor(
            estimators=[
                ('tree', HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05)),
                ('nn', DeepNeuralNet(epochs=200, lr=0.002))
            ],
            final_estimator=RidgeCV(), n_jobs=1
        )
        reg.fit(X_train_s, y_train[:, i])
        final_models.append(reg)
        y_pred_val[:, i] = reg.predict(X_val_s)
        
    joblib.dump(final_models, os.path.join(SAVE_DIR, 'models.pkl'))
    
    # 简洁评估报告
    final_r2 = r2_score(y_val, y_pred_val)
    final_mse = mean_squared_error(y_val, y_pred_val)
    print(f"✅ 完成 | R2: {final_r2:.4f} | MSE: {final_mse:.6f} | Path: {SAVE_DIR}")

if __name__ == '__main__':
    train_nonlinear_system()