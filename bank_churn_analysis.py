import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# ========== 字體設定 ==========
font_path = "C:/Windows/Fonts/msjh.ttc"
fontprop = font_manager.FontProperties(fname=font_path)

# ========== 輸出資料夾 ==========
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

# ========== 讀取資料 ==========
df = pd.read_csv("bank_churn.csv")

# ========== 資料檢查 ==========
print("資料概況：")
print(df.info())
print("\n缺失值：")
print(df.isnull().sum())

sns.set(style="whitegrid")

# ========== 圖表 ==========
# 年齡分布 vs 流失
plt.figure(figsize=(6,4))
sns.histplot(data=df, x="Age", hue="Churn", multiple="stack", bins=15)
plt.title("年齡分布 vs 流失", fontproperties=fontprop)
plt.savefig(os.path.join(output_dir, "age_vs_churn.png"))
plt.close()

# 帳戶餘額 vs 流失
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="Churn", y="Account_Balance")
plt.title("帳戶餘額 vs 流失", fontproperties=fontprop)
plt.savefig(os.path.join(output_dir, "balance_vs_churn.png"))
plt.close()

# 轉帳頻率 vs 流失
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="Churn", y="Transfer_Frequency")
plt.title("轉帳頻率 vs 流失", fontproperties=fontprop)
plt.savefig(os.path.join(output_dir, "transfer_vs_churn.png"))
plt.close()

# 投資習慣 vs 流失
plt.figure(figsize=(5,4))
sns.countplot(data=df, x="Investment_Flag", hue="Churn")
plt.title("投資習慣 vs 流失", fontproperties=fontprop)
plt.savefig(os.path.join(output_dir, "investment_vs_churn.png"))
plt.close()

# ========== 預測模型 ==========
X = df[["Age", "Account_Balance", "Transfer_Frequency", "Investment_Flag"]]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# ========== 輸出結果 ==========
with open("result.txt", "w", encoding="utf-8") as f:
    f.write(f"模型準確率: {accuracy * 100:.2f}%\n\n")
    f.write("混淆矩陣:\n")
    f.write(str(conf_mat) + "\n\n")
    f.write("分類報告:\n")
    f.write(report)

print(f"模型準確率: {accuracy * 100:.2f}%")
print("結果已輸出至 result.txt 和 images 資料夾")