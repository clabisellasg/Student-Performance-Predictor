"""
Student Performance Predictor - Main Blueprint
Predicts student grades and performance based on various academic and personal factors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    mean_squared_error, r2_score, mean_absolute_error
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA LOADING & EXPLORATION
# ============================================================================

def load_data(filepath):
    """Load the student performance dataset"""
    df = pd.read_csv(filepath)
    print(f"✓ Dataset loaded: {df.shape[0]} students, {df.shape[1]} features")
    return df


def explore_data(df):
    """Perform initial data exploration"""
    print("\n" + "="*60)
    print("DATA EXPLORATION")
    print("="*60)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nColumn Data Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    
    # Statistical summary
    print(f"\nNumerical Features Summary:\n{df.describe()}")
    
    # Grade distribution
    print(f"\nGrade Distribution:\n{df['Grade'].value_counts().sort_index()}")
    
    # Department distribution
    print(f"\nDepartment Distribution:\n{df['Department'].value_counts()}")
    
    return df


# ============================================================================
# 2. DATA PREPROCESSING & FEATURE ENGINEERING
# ============================================================================

def preprocess_data(df):
    """Clean and preprocess the dataset"""
    df = df.copy()
    
    print("\n" + "="*60)
    print("DATA PREPROCESSING")
    print("="*60)
    
    # Drop irrelevant columns for prediction
    columns_to_drop = ['Student_ID', 'First_Name', 'Last_Name', 'Email']
    df = df.drop(columns=columns_to_drop)
    print(f"\n✓ Dropped irrelevant columns: {columns_to_drop}")
    
    # Encode categorical variables
    categorical_cols = ['Gender', 'Department', 'Extracurricular_Activities', 
                        'Internet_Access_at_Home', 'Parent_Education_Level', 'Family_Income_Level']
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"✓ Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    print(f"\n✓ Categorical variables encoded")
    
    return df, label_encoders


# ============================================================================
# 3. FEATURE & TARGET PREPARATION
# ============================================================================

def prepare_features_targets(df):
    """Separate features and targets for analysis"""
    
    print("\n" + "="*60)
    print("FEATURE & TARGET PREPARATION")
    print("="*60)
    
    # Target 1: Grade Prediction (Classification)
    X = df.drop(columns=['Grade', 'Total_Score'])
    y_grade = df['Grade']
    
    # Target 2: Total Score Prediction (Regression)
    y_score = df['Total_Score']
    
    print(f"\n✓ Features: {X.shape}")
    print(f"✓ Feature columns: {list(X.columns)}")
    print(f"\n✓ Target (Grade Classification): {y_grade.shape}")
    print(f"✓ Target (Score Regression): {y_score.shape}")
    
    return X, y_grade, y_score


# ============================================================================
# 4. MODEL 1: GRADE PREDICTION (CLASSIFICATION)
# ============================================================================

def train_grade_classifier(X, y_grade):
    """Train a Random Forest classifier to predict student grades"""
    
    print("\n" + "="*60)
    print("MODEL 1: GRADE PREDICTION (Classification)")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_grade, test_size=0.2, random_state=42, stratify=y_grade
    )
    
    print(f"\n✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"✓ Features scaled")
    
    # Train Random Forest Classifier
    print(f"\n→ Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test_scaled)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✓ Model Accuracy: {accuracy:.4f}")
    
    print(f"\n📊 Classification Report:\n{classification_report(y_test, y_pred)}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n📈 Top 10 Important Features for Grade Prediction:")
    print(feature_importance.head(10).to_string(index=False))
    
    return clf, scaler, feature_importance, X_test_scaled, y_test


# ============================================================================
# 5. MODEL 2: TOTAL SCORE PREDICTION (REGRESSION)
# ============================================================================

def train_score_regressor(X, y_score):
    """Train a Gradient Boosting regressor to predict total scores"""
    
    print("\n" + "="*60)
    print("MODEL 2: TOTAL SCORE PREDICTION (Regression)")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_score, test_size=0.2, random_state=42
    )
    
    print(f"\n✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"✓ Features scaled")
    
    # Train Gradient Boosting Regressor
    print(f"\n→ Training Gradient Boosting Regressor...")
    regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = regressor.predict(X_test_scaled)
    
    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n✓ Model Performance:")
    print(f"  • Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  • Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  • R² Score: {r2:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': regressor.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n📈 Top 10 Important Features for Score Prediction:")
    print(feature_importance.head(10).to_string(index=False))
    
    return regressor, scaler, feature_importance, X_test_scaled, y_test


# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================

def create_visualizations(df, feature_importance_grade, feature_importance_score):
    """Create visualizations for analysis"""
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Grade Distribution
    ax1 = axes[0, 0]
    df['Grade'].value_counts().sort_index().plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Student Grade Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Grade')
    ax1.set_ylabel('Count')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Department Distribution
    ax2 = axes[0, 1]
    df['Department'].value_counts().plot(kind='bar', ax=ax2, color='lightcoral')
    ax2.set_title('Students by Department', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Department')
    ax2.set_ylabel('Count')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Feature Importance - Grade Prediction (Top 10)
    ax3 = axes[1, 0]
    top_features_grade = feature_importance_grade.head(10)
    ax3.barh(range(len(top_features_grade)), top_features_grade['Importance'], color='lightgreen')
    ax3.set_yticks(range(len(top_features_grade)))
    ax3.set_yticklabels(top_features_grade['Feature'])
    ax3.set_xlabel('Importance')
    ax3.set_title('Top 10 Features - Grade Prediction', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Feature Importance - Score Prediction (Top 10)
    ax4 = axes[1, 1]
    top_features_score = feature_importance_score.head(10)
    ax4.barh(range(len(top_features_score)), top_features_score['Importance'], color='lightyellow')
    ax4.set_yticks(range(len(top_features_score)))
    ax4.set_yticklabels(top_features_score['Feature'])
    ax4.set_xlabel('Importance')
    ax4.set_title('Top 10 Features - Score Prediction', fontsize=12, fontweight='bold')
    ax4.invert_yaxis()
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/analysis_visualization.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: results/analysis_visualization.png")
    plt.show()


# ============================================================================
# 7. SAMPLE PREDICTIONS
# ============================================================================

def make_sample_predictions(clf, regressor, scaler_clf, scaler_reg, X_test, y_test_grade, y_test_score, n_samples=5):
    """Make sample predictions on test data"""
    
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    # Select random samples
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    print(f"\n📋 Making {n_samples} sample predictions:\n")
    
    for i, idx in enumerate(indices, 1):
        sample = X_test[idx:idx+1]
        
        # Grade prediction
        sample_scaled_clf = scaler_clf.transform(sample)
        grade_pred = clf.predict(sample_scaled_clf)[0]
        
        # Score prediction
        sample_scaled_reg = scaler_reg.transform(sample)
        score_pred = regressor.predict(sample_scaled_reg)[0]
        
        print(f"  Sample {i}:")
        print(f"    • Predicted Grade: {grade_pred}")
        print(f"    • Actual Grade: {y_test_grade.iloc[idx]}")
        print(f"    • Predicted Score: {score_pred:.2f}")
        print(f"    • Actual Score: {y_test_score.iloc[idx]:.2f}\n")


# ============================================================================
# 8. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print(" "*15 + "STUDENT PERFORMANCE PREDICTOR")
    print("="*70)
    
    # Load and explore data
    df = load_data('../data/StudentPerformance.csv')
    df = explore_data(df)
    
    # Preprocess data
    df_processed, encoders = preprocess_data(df)
    
    # Prepare features and targets
    X, y_grade, y_score = prepare_features_targets(df_processed)
    
    # Train models
    clf, scaler_clf, importance_grade, X_test_clf, y_test_grade = train_grade_classifier(X, y_grade)
    regressor, scaler_reg, importance_score, X_test_reg, y_test_score = train_score_regressor(X, y_score)
    
    # Create visualizations
    create_visualizations(df, importance_grade, importance_score)
    
    # Make sample predictions
    make_sample_predictions(clf, regressor, scaler_clf, scaler_reg, X_test_clf, y_test_grade, y_test_score)
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print("\nNext Steps:")
    print("  • Evaluate model performance based on metrics")
    print("  • Tune hyperparameters for better accuracy")
    print("  • Deploy models to production")
    print("  • Create API endpoints for predictions")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()