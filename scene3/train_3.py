import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
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

def train_fashion_model():
    SAVE_DIR = MODEL_DIRS['scene3']
    
    # 1. 数据
    (train_img, train_lbl), (test_img, test_lbl) = tf.keras.datasets.fashion_mnist.load_data()
    train_img = train_img.reshape((-1, 28, 28, 1)) / 255.0
    test_img = test_img.reshape((-1, 28, 28, 1)) / 255.0

    # 2. 模型 (CNN + BN)
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
        layers.BatchNormalization(), layers.Activation('relu'), layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(), layers.Activation('relu'), layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(), layers.Activation('relu'),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 3. 训练
    callbacks = [
        ModelCheckpoint(os.path.join(SAVE_DIR, 'my_model.h5'), save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=2)
    ]

    history = model.fit(train_img, train_lbl, epochs=20, batch_size=128, 
                        validation_data=(test_img, test_lbl), callbacks=callbacks)
    
    print(f"🏆 Best Acc: {max(history.history['val_accuracy']):.4f}")

if __name__ == '__main__':
    train_fashion_model()