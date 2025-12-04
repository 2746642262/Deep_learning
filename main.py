import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import mysql.connector
import os
import joblib
import pickle
import torch
import torch.nn as nn
from tensorflow.keras.models import load_model
from sklearn.base import BaseEstimator, RegressorMixin
from config import MODEL_DIRS, DB_CONFIG

# --- PyTorch 类定义 (保持不变) ---
class DeepNeuralNet(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=8, output_dim=1, epochs=200, lr=0.002):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.device = torch.device('cpu') 
    
    def predict(self, X):
        if self.model is None: return np.zeros(len(X))
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            pred = self.model(X_t).cpu().numpy()
            if self.output_dim == 1:
                return pred.ravel()
            return pred

class DeepLearningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("深度学习综合预测系统")
        self.root.geometry("600x750")
        self.root.configure(bg="#f0f0f0")
        
        # 初始化数据库
        self.init_db()
        
        # 模型缓存
        self.models = {}
        self.preprocessors = {}
        self.selected_img = None
        
        # 加载静态资源
        try:
            with open(os.path.join(MODEL_DIRS['scene2'], 'scaler.pkl'), 'rb') as f:
                self.preprocessors['titanic_scaler'] = pickle.load(f)
        except:
            print("⚠️ Titanic Scaler not found.")

        # 构建UI
        self.create_widgets()

    def init_db(self):
        try:
            self.db = mysql.connector.connect(**DB_CONFIG)
            self.cursor = self.db.cursor()
            print("✓ Database connected")
        except Exception as e:
            messagebox.showerror("DB Error", f"Database connection failed:\n{e}")

    def create_widgets(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.build_flower_tab(notebook)
        self.build_fashion_tab(notebook)
        self.build_titanic_tab(notebook)
        self.build_nonlinear_tab(notebook)

    # --- 通用组件 ---
    def create_tab_base(self, notebook, title, header):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=title)
        
        main = tk.Frame(frame, bg="#f5f5f5")
        main.pack(fill='both', expand=True, padx=15, pady=15)
        
        tk.Label(main, text=header, font=("Arial", 16, "bold"), bg="#f5f5f5", fg="#1565C0").pack(pady=10)
        
        status_lbl = tk.Label(main, text="状态: 未加载", bg="white", fg="#555", relief="raised")
        status_lbl.pack(fill='x', pady=5)
        
        return main, status_lbl

    def create_btn(self, parent, text, cmd, color="#2196F3"):
        tk.Button(parent, text=text, command=cmd, bg=color, fg="#c52821", 
                 font=("Arial", 10), relief="flat", padx=15, pady=5).pack(side='left', padx=5)

    # --- 场景 1: 花卉 ---
    def build_flower_tab(self, notebook):
        main, self.lbl_flower_status = self.create_tab_base(notebook, "🌸 花卉识别", "花卉图像分类系统")
        
        btn_box = tk.Frame(main, bg="#f5f5f5")
        btn_box.pack(pady=10)
        self.create_btn(btn_box, "📥 加载模型", self.load_flower_model, "#FF9800")
        self.create_btn(btn_box, "📂 选择图片", lambda: self.select_image(self.lbl_flower_img, self.predict_flower), "#4CAF50")
        
        # 图片显示区域 (初始带有占位符)
        self.lbl_flower_img = tk.Label(main, bg="#e0e0e0", text="[ 图片预览区域 ]", width=20, height=20)
        self.lbl_flower_img.pack(pady=10)
        
        self.lbl_flower_res = tk.Label(main, text="", font=("Arial", 18, "bold"), bg="#f5f5f5", fg="#E91E63")
        self.lbl_flower_res.pack()

    def load_flower_model(self):
        try:
            path = os.path.join(MODEL_DIRS['scene1'], 'flowers.h5')
            self.models['flower'] = load_model(path)
            self.preprocessors['flower_le'] = joblib.load(os.path.join(MODEL_DIRS['scene1'], 'label_encoder.pkl'))
            self.lbl_flower_status.config(text="✓ 花卉模型已就绪", fg="green")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def predict_flower(self):
        if 'flower' not in self.models: 
            self.lbl_flower_res.config(text="请先加载模型！", fg="red")
            return
        try:
            img = cv2.imread(self.selected_img)
            # 预测时必须 resize 到模型需要的 150x150
            img_input = cv2.resize(img, (150, 150)) / 255.0
            res = self.models['flower'].predict(np.expand_dims(img_input, 0), verbose=0)
            
            en_label = self.preprocessors['flower_le'].classes_[np.argmax(res)]
            zh_map = {"daisy":"雏菊", "rose":"玫瑰", "tulip":"郁金香", "sunflower":"向日葵", "dandelion":"蒲公英"}
            pred = zh_map.get(en_label, en_label)
            
            self.lbl_flower_res.config(text=f"识别结果: {pred}", fg="#E91E63")
            self.save_to_db("Flower", self.selected_img, pred)
        except Exception as e:
            print(e)

    # --- 场景 2: Titanic ---
    def build_titanic_tab(self, notebook):
        main, self.lbl_titanic_status = self.create_tab_base(notebook, "🚢 Titanic", "生存概率预测")
        
        btn_box = tk.Frame(main, bg="#f5f5f5")
        btn_box.pack(pady=10)
        self.create_btn(btn_box, "📥 加载模型", self.load_titanic_model, "#FF9800")
        self.create_btn(btn_box, "🎲 计算概率", self.predict_titanic, "#673AB7")

        form = tk.Frame(main, bg="#f5f5f5")
        form.pack(pady=10)
        self.titanic_entries = {}
        fields = [
            ("舱位 (1-3)", "pclass", ["1","2","3"]),
            ("性别", "sex", ["Male", "Female"]),
            ("年龄", "age", None),
            ("兄弟姐妹", "sibsp", None),
            ("父母子女", "parch", None),
            ("票价", "fare", None),
            ("港口", "embarked", ["S","C","Q"])
        ]
        
        for i, (txt, key, opts) in enumerate(fields):
            tk.Label(form, text=txt, bg="#f5f5f5", width=10, anchor='e').grid(row=i, column=0, pady=5)
            if opts:
                w = ttk.Combobox(form, values=opts, state="readonly", width=18)
            else:
                w = tk.Entry(form, width=20)
            w.grid(row=i, column=1, padx=10)
            self.titanic_entries[key] = w
            
        self.lbl_titanic_res = tk.Label(main, text="", font=("Arial", 16, "bold"), bg="#f5f5f5", fg="#E91E63")
        self.lbl_titanic_res.pack()

    def load_titanic_model(self):
        try:
            self.models['titanic'] = load_model(os.path.join(MODEL_DIRS['scene2'], 'taitanic.h5'))
            self.lbl_titanic_status.config(text="✓ Titanic模型已就绪", fg="green")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def predict_titanic(self):
        if 'titanic' not in self.models: 
            self.lbl_titanic_res.config(text="请先加载模型！", fg="red")
            return
        d = self.titanic_entries
        try:
            feats = [
                int(d['pclass'].get()),
                0 if d['sex'].get() == "Male" else 1,
                float(d['age'].get()),
                int(d['sibsp'].get()),
                int(d['parch'].get()),
                float(d['fare'].get()),
                ["S","C","Q"].index(d['embarked'].get())
            ]
            feats_scaled = self.preprocessors['titanic_scaler'].transform([feats])
            prob = self.models['titanic'].predict(feats_scaled, verbose=0)[0][0]
            
            self.lbl_titanic_res.config(text=f"生存概率: {prob:.2%}", fg="#E91E63")
            self.cursor.execute(
                "INSERT INTO Titanic (pclass, sex, age, sibsp, parch, fare, embarked, survival_probability) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (*feats, float(prob))
            )
            self.db.commit()
            print("DB Saved.")
        except Exception as e:
            messagebox.showerror("Input Error", str(e))

    # --- 场景 3: Fashion ---
    def build_fashion_tab(self, notebook):
        main, self.lbl_fashion_status = self.create_tab_base(notebook, "👕 服装分类", "时尚单品识别")
        
        btn_box = tk.Frame(main, bg="#f5f5f5")
        btn_box.pack(pady=10)
        self.create_btn(btn_box, "📥 加载模型", self.load_fashion_model, "#FF9800")
        self.create_btn(btn_box, "📂 选择图片", lambda: self.select_image(self.lbl_fashion_img, self.predict_fashion), "#4CAF50")
        
        # 图片显示区域
        self.lbl_fashion_img = tk.Label(main, bg="#e0e0e0", text="[ 图片预览区域 ]", width=20, height=20)
        self.lbl_fashion_img.pack(pady=10)
        
        self.lbl_fashion_res = tk.Label(main, text="", font=("Arial", 18, "bold"), bg="#f5f5f5", fg="#E91E63")
        self.lbl_fashion_res.pack()
        
    def load_fashion_model(self):
        try:
            self.models['fashion'] = load_model(os.path.join(MODEL_DIRS['scene3'], 'my_model.h5'))
            self.lbl_fashion_status.config(text="✓ 服装模型已就绪", fg="green")
            self.class_names = ['T恤', '裤子', '套头衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '包', '靴子']
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def predict_fashion(self):
        if 'fashion' not in self.models: 
            self.lbl_fashion_res.config(text="请先加载模型！", fg="red")
            return
        try:
            img = cv2.imread(self.selected_img, cv2.IMREAD_GRAYSCALE)
            img_input = cv2.resize(img, (28, 28)) / 255.0
            res = self.models['fashion'].predict(img_input.reshape(1, 28, 28, 1), verbose=0)
            pred = self.class_names[np.argmax(res)]
            self.lbl_fashion_res.config(text=f"分类: {pred}", fg="#E91E63")
            self.save_to_db("Clothes", self.selected_img, pred)
        except Exception as e:
            print(e)

    # --- 场景 4: Nonlinear ---
    def build_nonlinear_tab(self, notebook):
        main, self.lbl_nonlinear_status = self.create_tab_base(notebook, "📊 非线性", "复杂系统预测 (8->3)")
        
        btn_box = tk.Frame(main, bg="#f5f5f5")
        btn_box.pack(pady=10)
        self.create_btn(btn_box, "📥 加载模型", self.load_nonlinear_model, "#FF9800")
        self.create_btn(btn_box, "🔮 预测", self.predict_nonlinear, "#4CAF50")
        
        form = tk.Frame(main, bg="#f5f5f5")
        form.pack()
        self.nl_entries = []
        for i in range(8):
            tk.Label(form, text=f"特征 {i+1}", bg="#f5f5f5").grid(row=i//2, column=(i%2)*2, padx=5, pady=5)
            e = tk.Entry(form, width=10)
            e.grid(row=i//2, column=(i%2)*2+1, padx=5)
            self.nl_entries.append(e)
            
        self.lbl_nl_res = tk.Label(main, text="", font=("Arial", 12), bg="#f5f5f5", fg="blue")
        self.lbl_nl_res.pack(pady=10)

    def load_nonlinear_model(self):
        try:
            base = MODEL_DIRS['scene4']
            self.preprocessors['nl_poly'] = joblib.load(os.path.join(base, 'poly.pkl'))
            self.preprocessors['nl_scaler'] = joblib.load(os.path.join(base, 'scaler_x.pkl'))
            self.models['nonlinear'] = joblib.load(os.path.join(base, 'models.pkl'))
            self.lbl_nonlinear_status.config(text="✓ 混合模型堆叠已加载", fg="green")
        except Exception as e:
            print(f"Loading Error: {e}")
            messagebox.showerror("Error", str(e))

    def predict_nonlinear(self):
        if 'nonlinear' not in self.models: 
            self.lbl_nl_res.config(text="请先加载模型！", fg="red")
            return
        try:
            inputs = [float(e.get()) for e in self.nl_entries]
            
            feats = self.preprocessors['nl_poly'].transform([inputs])
            feats = self.preprocessors['nl_scaler'].transform(feats)
            
            preds = []
            for model in self.models['nonlinear']:
                raw_pred = model.predict(feats)[0]
                preds.append(raw_pred)
            
            self.lbl_nl_res.config(text=f"Y1: {preds[0]:.3f} | Y2: {preds[1]:.3f} | Y3: {preds[2]:.3f}", fg="#4CAF50")
            
            sql = "INSERT INTO Nonlinear (feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,prediction1,prediction2,prediction3) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            self.cursor.execute(sql, (*inputs, *preds))
            self.db.commit()
            print("Nonlinear Prediction Saved to DB.")
            
        except Exception as e:
            print(f"Prediction Error: {e}")
            messagebox.showerror("Calc Error", str(e))

    # ==========================================
    # 🔥 关键修改：优化的图片显示逻辑
    # ==========================================
    def select_image(self, lbl_widget, callback):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")])
        if path:
            self.selected_img = path
            try:
                # 1. 打开图片
                pil_image = Image.open(path)
                
                # 2. 智能缩放 (Thumbnail): 保持长宽比，最大限制为 300x300
                # 这解决了"图片太小"和"被裁剪/变形"的问题
                display_size = (300, 300) 
                pil_image.thumbnail(display_size, Image.LANCZOS)
                
                # 3. 转换为 Tkinter 对象
                photo = ImageTk.PhotoImage(pil_image)
                
                # 4. 更新 Label
                # width=0, height=0 告诉 Label 自动适应图片大小，而不是使用字符单位
                lbl_widget.config(image=photo, width=0, height=0)
                lbl_widget.image = photo # 必须保持引用，否则图片会消失
                
                # 5. 自动触发预测
                callback() 
            except Exception as e:
                print(f"Image Load Error: {e}")
                messagebox.showerror("Error", "无法加载图片")
    
    def save_to_db(self, table, fname, cat):
        try:
            self.cursor.execute(f"INSERT INTO {table} (filename, category) VALUES (%s, %s)", (fname, cat))
            self.db.commit()
            print(f"Saved to {table}")
        except: pass

if __name__ == "__main__":
    root = tk.Tk()
    app = DeepLearningApp(root)
    root.mainloop()