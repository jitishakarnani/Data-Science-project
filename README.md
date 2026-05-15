# 🔍 Sensor Anomaly Detection

## 📌 Overview
Machine Learning project to detect anomalies in sensor readings.
Built for CEIP_DS_JECRC Kaggle Competition.

**Final Score: 0.805 | Rank: Top 25**

---

## 📂 Project Structure
anomaly-detection/
├── anomaly_detection.ipynb  # Main notebook
├── train.parquet            # Training data
├── test.parquet             # Test data
├── sample_submission.parquet # Submission format
└── submission.csv           # Final predictions

---

## 📊 Dataset
| Detail | Info |
|---|---|
| Train Size | 1.6 Million rows |
| Test Size | 409,856 rows |
| Features | X1, X2, X3, X4, X5, Date |
| Target | 0 = Normal, 1 = Anomaly |
| Challenge | Highly imbalanced (0.86% anomalies) |

---

## 🔧 Feature Engineering
- **Log Transform** — X3, X4 were on exponential scale
- **Date Features** — month, day of week, quarter
- **Interaction Features** — sensor relationships
- **Polynomial Features** — squared terms
- Total: **19 features** created

---

## 🤖 Models Used

### Classical Models
| Model | F1 Score |
|---|---|
| Logistic Regression | ~0.35 |
| K-Nearest Neighbors | ~0.30 |
| Decision Tree | ~0.35 |

### Advanced Models
| Model | F1 Score |
|---|---|
| Random Forest | ~0.55 |
| XGBoost | ~0.65 |
| LightGBM | ~0.80 ✅ |

---

## 📈 Results
| Submission | Approach | Score |
|---|---|---|
| 1 | Logistic Regression | 0.237 |
| 2 | LR + Threshold Tuning | 0.443 |
| 3 | LightGBM | 0.671 |
| 4 | LGBM + XGBoost Ensemble | 0.725 |
| 5 | Ensemble + More Features | 0.754 |
| 6 | Final Ensemble | **0.805** ✅ |

---

## 🛠️ Tech Stack
- Python 3.12
- Pandas, NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Matplotlib, Seaborn

---

## 🚀 How to Run
1. Clone the repository
git clone https://github.com/yourusername/anomaly-detection

2. Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn pyarrow jupyter

3. Run the notebook
jupyter notebook anomaly_detection.ipynb

---

## 💡 Key Learnings
1. Class imbalance needs special handling — `scale_pos_weight`
2. Feature engineering is crucial — log transform boosted correlation 0.04 → 0.37
3. F1 Score > Accuracy for imbalanced datasets
4. Threshold tuning significantly improves F1 score
5. Ensemble models outperform classical models

---

## 👩‍💻 Author
**Jitisha Karnani**
MTech CS
