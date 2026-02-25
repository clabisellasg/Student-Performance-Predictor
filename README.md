# Student Performance Predictor 🎓

A machine learning project that predicts student academic performance based on various academic, personal, and socioeconomic factors.

## 📊 Project Overview

This project uses student data with 22+ features including:
- **Academic Performance**: Attendance, Midterm/Final scores, Assignments, Quizzes, Projects
- **Personal Factors**: Age, Gender, Stress Level, Sleep Hours
- **Socioeconomic**: Family Income, Parent Education, Internet Access
- **Engagement**: Extracurricular Activities, Participation

### Prediction Models

1. **Grade Classifier** - Predicts student letter grades (A, B, C, D, F)
   - Algorithm: Random Forest Classification
   - Metric: Accuracy Score

2. **Score Regressor** - Predicts total performance score (0-100)
   - Algorithm: Gradient Boosting Regression
   - Metrics: MAE, RMSE, R² Score

## 📁 Project Structure

```
Student-Performance-Predictor/
├── data/
│   └── StudentPerformance.csv (5000+ student records)
├── notebooks/
│   └── (for exploratory analysis)
├── src/
│   └── main.py (main pipeline)
├── results/
│   └── analysis_visualization.png (generated charts)
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
```bash
cd src
python main.py
```

### 3. Output Includes
- ✅ Data exploration and statistics
- ✅ Model training and evaluation metrics
- ✅ Feature importance analysis
- ✅ Visualizations (saved to `results/`)
- ✅ Sample predictions on test data

## 📈 Pipeline Steps

### 1. **Data Loading & Exploration**
   - Load StudentPerformance.csv
   - Display dataset statistics and distributions
   - Identify data patterns

### 2. **Data Preprocessing**
   - Remove irrelevant columns (ID, Names, Email)
   - Encode categorical variables (Gender, Department, etc.)
   - Handle missing values

### 3. **Feature Engineering**
   - Separate features and targets
   - Create feature matrix (X) and target variables (y)

### 4. **Model Training**
   - Split data (80% train, 20% test)
   - Scale features using StandardScaler
   - Train Random Forest Classifier (grades)
   - Train Gradient Boosting Regressor (scores)

### 5. **Evaluation**
   - Classification Report (precision, recall, F1-score)
   - Confusion Matrix
   - Accuracy, MAE, RMSE, R² metrics

### 6. **Analysis & Visualization**
   - Feature importance rankings
   - Grade and department distributions
   - Performance charts

### 7. **Predictions**
   - Sample predictions on test set
   - Compare predicted vs actual values

## 🔧 Model Features

### Key Hyperparameters
- **Random Forest**: 100 estimators, random_state=42
- **Gradient Boosting**: 100 estimators, learning_rate=0.1
- **Train/Test Split**: 80/20 with stratification for grades
- **Feature Scaling**: StandardScaler normalization

### Feature Encoding
| Column | Encoding Type |
|--------|---------------|
| Gender | Label Encoding |
| Department | Label Encoding |
| Categorical Vars | Label Encoding |
| Numerical Vars | StandardScaler |

## 📊 Key Insights

The model identifies relationships between:
- Study effort and academic performance
- Attendance patterns and grades
- Socioeconomic factors (family income, parent education)
- Personal factors (sleep, stress, age)
- Engagement (extracurricular activities, participation)

## 🎯 Future Enhancements

- [ ] Hyperparameter tuning (GridSearch, RandomSearch)
- [ ] Cross-validation for robust evaluation
- [ ] Handle class imbalance in grades
- [ ] Feature selection (PCA, correlation analysis)
- [ ] Deploy as REST API (Flask/FastAPI)
- [ ] Docker containerization
- [ ] Real-time prediction interface
- [ ] Student intervention recommendations

## 📝 Notes

- Dataset has 5000+ student records with 22 features
- Grades are imbalanced (fewer A's and F's, more C's and D's)
- Features show strong correlation with academic performance
- Models serve as baselines for production optimizations

## 📧 Contact

For questions or improvements, feel free to contribute!

---

**Status**: ✅ Working Blueprint | **Version**: 1.0