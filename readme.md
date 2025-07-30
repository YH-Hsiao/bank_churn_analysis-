# 銀行客戶流失分析 (Bank Churn Analysis)

## 專案目標
分析銀行客戶流失的主要影響因素，並建立預測模型，協助銀行制定挽留策略。

---

## 使用技術
- Python (pandas, matplotlib, seaborn, scikit-learn)
- 資料標準化 (StandardScaler)
- Logistic Regression (類別平衡)

---

## 分析重點
1. 年齡較大的客戶流失比例略高
2. 低餘額客戶流失率較高
3. 沒有投資習慣的客戶更容易流失
4. 轉帳頻率差異不顯著

---

## 預測模型
- 模型：Logistic Regression (含標準化與類別權重調整)
- 模型準確率：約 56%

---

## 產出成果
- 圖表輸出於 `images/` 資料夾：
  - 年齡分布 vs 流失 (`age_vs_churn.png`)
  - 帳戶餘額 vs 流失 (`balance_vs_churn.png`)
  - 轉帳頻率 vs 流失 (`transfer_vs_churn.png`)
  - 投資習慣 vs 流失 (`investment_vs_churn.png`)
- 分析結果輸出於 `result.txt`

---

## 執行方式
1. 安裝所需套件
```bash
pip install pandas matplotlib seaborn scikit-learn
```
2. 執行分析程式
```bash
python bank_churn_analysis.py
```
3. 圖表與結果將自動輸出至 `images/` 資料夾與 `result.txt`

---

## 商業建議
1. 針對低餘額、低交易頻率客戶推出優惠活動  
2. 推廣投資產品以提高客戶黏著度  
3. 提供高齡客戶專屬服務與關懷
