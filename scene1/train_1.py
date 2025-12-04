import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import pathlib
import sys
import platform

# 设置项目根目录以导入配置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
from app.config import MODEL_DIRS

if platform.system() == 'Darwin' and 'arm' in platform.machine():
    from tensorflow.keras.optimizers.legacy import Adam
else:
    from tensorflow.keras.optimizers import Adam

def train_flower_model():
    SAVE_DIR = MODEL_DIRS['scene1']
    IMG_SIZE = (150, 150)
    BATCH_SIZE = 32

    # 1. 数据准备
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    # 2. 数据增强 (训练集)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training'
    )
    val_gen = train_datagen.flow_from_directory(
        data_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation'
    )

    # 3. 保存标签
    le = LabelEncoder()
    le.fit(sorted(list(train_gen.class_indices.keys())))
    joblib.dump(le, os.path.join(SAVE_DIR, 'label_encoder.pkl'))

    # 4. 构建模型 (DenseNet121 迁移学习)
    base_model = DenseNet121(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.Dense(5, activation='softmax')
    ])

    # 5. 两阶段训练策略
    # 阶段一：冻结基座，训练头部
    base_model.trainable = False
    
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
        ReduceLROnPlateau(factor=0.2, patience=2)
    ]
    
    print("\n>>> Phase 1: Training Head...")
    model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=callbacks)

    # 阶段二：解冻微调
    print("\n>>> Phase 2: Fine-tuning...")
    base_model.trainable = True
    
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(train_gen, validation_data=val_gen, epochs=20, callbacks=callbacks)

    # 6. 保存
    acc = history.history['val_accuracy'][-1]
    print(f"🏆 Final Accuracy: {acc:.2%}")
    model.save(os.path.join(SAVE_DIR, 'flowers.h5'))

if __name__ == '__main__':
    train_flower_model()