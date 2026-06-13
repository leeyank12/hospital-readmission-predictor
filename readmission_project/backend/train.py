import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.preprocess import load_and_clean, engineer_features
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

def train(data_path='data/diabetic_data.csv'):
    
    print("Loading and cleaning data...")
    df = load_and_clean(data_path)
    
    print("Engineering features...")
    X, y, features = engineer_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    
    # Model 1 — Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    print(f"Logistic Regression AUC: {lr_auc:.3f}")
    
    # Model 2 — Random Forest with GridSearch
    print("\nTraining Random Forest with GridSearchCV...")
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [4, 6, 8]
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params, cv=3, scoring='roc_auc', n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    rf_auc = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])
    print(f"Random Forest AUC: {rf_auc:.3f}")
    print(f"Best RF params: {rf_grid.best_params_}")
    
    # Model 3 — Gradient Boosting
    print("\nTraining Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    gb_auc = roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1])
    print(f"Gradient Boosting AUC: {gb_auc:.3f}")
    
    # Pick best model
    results = [
        ('Logistic Regression', lr, lr_auc),
        ('Random Forest', best_rf, rf_auc),
        ('Gradient Boosting', gb, gb_auc)
    ]
    best_name, best_model, best_auc = max(results, key=lambda x: x[2])
    print(f"\nBest Model: {best_name} — AUC: {best_auc:.3f}")
    
    # Evaluation report
    preds = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    # Save model and features
    os.makedirs('models', exist_ok=True)
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('models/features.pkl', 'wb') as f:
        pickle.dump(features, f)
    
    print("\nModel saved to models/model.pkl")
    print("Features saved to models/features.pkl")
    
    return best_model, features, X_test, y_test

if __name__ == '__main__':
    train()