import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
import sys
import platform
system_os = platform.system()
machine = platform.machine()
if system_os == 'Darwin' and 'arm' in machine:
    from tensorflow.keras.optimizers.legacy import Adam
else:
    from tensorflow.keras.optimizers import Adam


# 导入配置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
from app.config import MODEL_DIRS
# ---------------

def train_titanic_model():
    SAVE_DIR = MODEL_DIRS['scene2']
    
    # 1. 数据预处理
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    
    # 特征工程 (保持与前端一致的逻辑)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[features].values
    y = df['Survived'].values

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    with open(os.path.join(SAVE_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 2. 模型构建 (宽深网络 + BN)
    model = Sequential([
        Dense(64, input_dim=7), BatchNormalization(), tf.keras.layers.Activation('relu'), Dropout(0.3),
        Dense(32), BatchNormalization(), tf.keras.layers.Activation('relu'), Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # 3. 训练
    checkpoint = ModelCheckpoint(
        os.path.join(SAVE_DIR, 'taitanic.h5'), 
        monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
    )
    
    model.fit(
        X_train, y_train, 
        epochs=200, batch_size=32, validation_data=(X_test, y_test),
        callbacks=[checkpoint, EarlyStopping(patience=20, restore_best_weights=True)],
        verbose=0 # 减少日志输出
    )
    
    print("✅ Model trained and saved.")

if __name__ == '__main__':
    train_titanic_model()