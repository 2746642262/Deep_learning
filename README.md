这是一份为您定制的 **README.md** 文档。它完全贴合您的项目目录结构、技术栈（Mac M1 加速、Tkinter GUI、MySQL）以及团队分工。

您可以直接将以下内容保存为项目根目录下的 `README.md` 文件。

---

# 深度学习综合预测系统 (Deep Learning Integrated Prediction System)

![Python](https://img.shields.io/badge/Python-3.9-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange) ![PyTorch](https://img.shields.io/badge/PyTorch-MPS-red) ![Platform](https://img.shields.io/badge/Platform-macOS%20M1%2FWindows-lightgrey)

## 📖 项目简介

本项目是一个基于深度学习的综合性预测平台，集成了计算机视觉（图像分类）和数据挖掘（结构化数据回归/分类）领域的四大核心应用场景。系统采用分层架构设计，前端基于 Tkinter 实现可视化交互，后端集成 TensorFlow 和 PyTorch 双框架，并针对 **Apple Silicon (M1/M2/M3)** 芯片进行了 Metal/MPS 硬件加速优化。

项目支持全链路业务流程：**模型训练 -> 动态加载 -> 实时推理 -> 数据库持久化 -> 自动化测试**。

---

## ✨ 核心功能 (四大场景)

| 场景 | 任务类型 | 模型架构 | 性能指标 | 描述 |
| :--- | :--- | :--- | :--- | :--- |
| **🌸 场景 1** | 花卉图像分类 | **DenseNet121** (迁移学习) | Acc > 90% | 基于 ImageNet 预训练，识别 5 种常见花卉。 |
| **🚢 场景 2** | Titanic 生存预测 | **Wide & Deep MLP** | Acc > 80% | 结合 BN 层与 Dropout，预测乘客生存概率。 |
| **👕 场景 3** | 时尚服饰分类 | **Custom CNN** | Acc > 92% | 针对 Fashion-MNIST 的 3 层卷积神经网络。 |
| **📊 场景 4** | 非线性系统回归 | **Stacking** (NN + Tree) | $R^2$ > 0.78 | 解决多输入多输出(MIMO)复杂非线性拟合问题。 |

---

## 📂 项目结构

```text
Project_Root/
├── app/
│   ├── config.py            # [配置] 全局路径与数据库配置中心
│   ├── db.py                # [工具] 数据库初始化脚本
│   ├── main.py              # [入口] GUI主程序
│   ├── analysis_report.py   # [分析] 可视化图表生成脚本
│   ├── system_test.py       # [测试] 自动化综合测试套件
│   ├── scene1/              # [模块] 花卉识别 (train_1.py + models/)
│   ├── scene2/              # [模块] Titanic预测 (train_2.py + models/)
│   ├── scene3/              # [模块] 服装分类 (train_3.py + models/)
│   └── scene4/              # [模块] 非线性回归 (train_4.py + data/ + models/)
├── requirements.txt         # 依赖库清单
└── README.md                # 项目说明文档
```

---

## 🛠️ 安装与配置

### 1. 环境准备
推荐使用 Anaconda 创建 Python 3.9 环境（**注意：必须使用 Python 3.9 以兼容 TensorFlow 2.13**）。

```bash
conda create -n tf-gpu python=3.9
conda activate tf-gpu
```

### 2. 安装依赖
根据您的操作系统选择安装方式。

**macOS (Apple Silicon M1/M2):**
```bash
# 关键：锁定 numpy 版本以防冲突
pip install -r requirements.txt
pip install "numpy==1.24.3" 
```

**Windows:**
请确保安装了对应 CUDA 版本的 PyTorch（如果有 N 卡），否则使用 CPU 版本。

### 3. 数据库配置
1.  确保本地已安装 **MySQL 8.0+**。
2.  修改 `app/config.py` 中的数据库连接信息：
    ```python
    DB_CONFIG = {
        "host": "localhost",
        "user": "root",
        "password": "YOUR_PASSWORD",  # 修改为您自己的密码
        "database": "Deep_learning"
    }
    ```
3.  运行初始化脚本创建数据库和表：
    ```bash
    python app/db.py
    ```

---

## 🚀 快速开始

### 1. 启动 GUI 系统
进入项目根目录，运行主程序：
```bash
python app/main.py
```
*   **操作**：选择对应标签页 -> 点击“加载模型” -> 选择图片/输入数据 -> 查看预测结果（结果会自动写入数据库）。

### 2. 模型训练 (可选)
如果需要重新训练模型，运行对应场景的脚本：
```bash
# 例如训练场景 4
python app/scene4/train_4.py
```
*   训练完成后，模型文件会自动保存到 `app/sceneX/models/` 目录。

### 3. 自动化测试
运行全链路测试脚本，验证模型精度和业务功能：
```bash
python app/system_test.py
```
*   输出结果将包含：✅ 性能测试 (Pass/Fail) + ✅ 功能测试 (DB验证) + ⏱️ 推理耗时。

### 4. 生成分析报告
生成混淆矩阵、ROC 曲线、残差分布图等：
```bash
python app/analysis_report.py
# 或在 Jupyter Notebook 中运行 app/analysis_report.ipynb
```

---

## 👥 团队分工

*   **(项目经理)**：系统架构设计、GUI 开发、数据库设计、统筹整合。
*   **(产品经理)**：花卉识别模块 (Scene 1) 算法实现与文档。
*   **(软件工程师)**：Titanic 预测模块 (Scene 2) 算法实现与文档。
*   **(系统工程师)**：服装识别模块 (Scene 3) 算法实现与文档。
*   **(UCD工程师)**：非线性回归模块 (Scene 4) 算法实现与文档。
*   **(测试工程师)**：系统测试策略制定、自动化测试脚本编写。

---

## ⚠️ 注意事项

1.  **路径问题**：所有脚本内部已配置动态路径回溯，请**在项目根目录下**运行脚本（如 `python app/main.py`），尽量避免进入子目录运行。
2.  **M1 兼容性**：代码中包含针对 `mps` (Metal Performance Shaders) 的自动检测逻辑。如果迁移到 Windows，代码会自动切换到 `cuda` 或 `cpu` 模式，无需修改。
3.  **NumPy 版本**：TensorFlow 2.13 与 NumPy 2.x 不兼容，请务必保持 `numpy==1.24.3`。

---

**© 2025 Deep Learning Team. All Rights Reserved.**
