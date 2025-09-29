# Logistic Regression Classification

## 📌 Task Overview

This project implements **binary classification** using **Logistic Regression** on the **Breast Cancer Wisconsin Dataset**.
The objective is to build, train, and evaluate a logistic regression model for detecting whether a tumor is **benign (0)** or **malignant (1)**.

---

## ⚙️ Tools & Libraries

* Python 3.x
* Pandas
* NumPy
* Scikit-learn
* Matplotlib / Seaborn

---

## 📂 Project Structure

```
Logistic-Regression-Classification/
│── data/                 # Dataset (CSV file)
│── src/
│   └── logistic_regression.py
│── outputs/
│   ├── confusion_matrix.png
│   └── roc_curve.png
│── requirements.txt
│── README.md
```

---

## 🚀 How to Run

1. Clone this repository:

   ```bash
   git clone <repo_url>
   cd Logistic-Regression-Classification
   ```

2. Create virtual environment & install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate   # for Linux/Mac
   venv\Scripts\activate      # for Windows
   pip install -r requirements.txt
   ```

3. Run the project:

   ```bash
   python src/logistic_regression.py
   ```

---

## 📊 Model Performance

* **Accuracy:** ~96%
* **ROC-AUC Score:** 0.996
* **Confusion Matrix (threshold=0.6):**

  ```
  [[72  0]
   [ 4 38]]
  ```

✅ The model performs very well with high precision, recall, and AUC score.

---

## 🧾 What You’ll Learn

* Difference between **Linear & Logistic Regression**
* How to use the **Sigmoid function** for probability prediction
* Evaluation metrics: **Confusion Matrix, Precision, Recall, ROC-AUC**
* Effect of **threshold tuning** on classification results

---

## 📎 Dataset

* **Breast Cancer Wisconsin Dataset** (from `sklearn.datasets.load_breast_cancer`)
* Features: 30
* Target: Diagnosis (`0 = Benign`, `1 = Malignant`)

---
