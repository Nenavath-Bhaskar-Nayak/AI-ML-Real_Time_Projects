"""
Student Performance ML Pipeline - Training and Evaluation Script
Executes full model exploration, hyperparameter tuning, metrics export, and model serialization.
"""
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'raw', 'student_performance.csv')
    models_dir = os.path.join(base_dir, 'models')
    reports_dir = os.path.join(base_dir, 'reports')

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("Loading data from:", data_path)
    df = pd.read_csv(data_path)

    numeric_features = ['Age', 'StudyHours', 'Attendance', 'AssignmentScore', 'PreviousExamScore', 'PracticeTestScore']
    categorical_features = ['Gender', 'Department']
    feature_cols = numeric_features + categorical_features

    X = df[feature_cols]
    y = df['Pass']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, numeric_features),
        ('cat', cat_pipeline, categorical_features)
    ])

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    comparison_records = []

    print("\nEvaluating baseline models...")
    for name, model in models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=1)
        pipe.fit(X_train, y_train)
        
        train_pred = pipe.predict(X_train)
        test_pred = pipe.predict(X_test)
        test_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, 'predict_proba') else None
        
        acc_train = accuracy_score(y_train, train_pred)
        acc_test = accuracy_score(y_test, test_pred)
        prec = precision_score(y_test, test_pred, zero_division=0)
        rec = recall_score(y_test, test_pred, zero_division=0)
        f1 = f1_score(y_test, test_pred, zero_division=0)
        auc = roc_auc_score(y_test, test_proba) if test_proba is not None else np.nan
        
        comparison_records.append({
            'Model': name,
            'Train_Accuracy': round(acc_train, 4),
            'Test_Accuracy': round(acc_test, 4),
            'CV_Mean_Accuracy': round(cv_scores.mean(), 4),
            'CV_Std_Accuracy': round(cv_scores.std(), 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1_Score': round(f1, 4),
            'ROC_AUC': round(auc, 4)
        })

    comparison_df = pd.DataFrame(comparison_records).sort_values(by='F1_Score', ascending=False)
    print("\nModel Comparison Table:")
    print(comparison_df.to_string(index=False))

    print("\nRunning Hyperparameter Optimization for Random Forest...")
    param_grid = {
        'classifier__n_estimators': [50, 100, 150],
        'classifier__max_depth': [3, 5, 8, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }

    rf_base_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    grid_search = GridSearchCV(
        rf_base_pipe,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=1
    )
    grid_search.fit(X_train, y_train)

    print("\nBest Parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Final Evaluation
    final_test_pred = best_model.predict(X_test)
    final_test_proba = best_model.predict_proba(X_test)[:, 1]
    print(f"Optimized Random Forest Test Accuracy: {accuracy_score(y_test, final_test_pred):.4f}")
    print(f"Optimized Random Forest Test F1:       {f1_score(y_test, final_test_pred):.4f}")
    print(f"Optimized Random Forest ROC-AUC:       {roc_auc_score(y_test, final_test_proba):.4f}")

    # Extract Feature Importances
    preprocessor_fitted = best_model.named_steps['preprocessor']
    cat_encoder = preprocessor_fitted.named_transformers_['cat'].named_steps['encoder']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + cat_feature_names

    importances = best_model.named_steps['classifier'].feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': np.round(importances, 5)
    }).sort_values(by='Importance', ascending=False)

    print("\nFeature Importances:")
    print(feat_imp_df.to_string(index=False))

    # Save to CSV and PKL
    model_comp_path = os.path.join(reports_dir, 'model_comparison.csv')
    feat_imp_path = os.path.join(reports_dir, 'feature_importance.csv')
    model_save_path = os.path.join(models_dir, 'student_pass_model.pkl')

    comparison_df.to_csv(model_comp_path, index=False)
    feat_imp_df.to_csv(feat_imp_path, index=False)
    joblib.dump(best_model, model_save_path)

    print(f"\nSaved comparison report to: {model_comp_path}")
    print(f"Saved feature importance report to: {feat_imp_path}")
    print(f"Saved trained pipeline model to: {model_save_path}")

if __name__ == '__main__':
    main()
