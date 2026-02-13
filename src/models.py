import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import MODEL_PARAMS, TEST_SIZE, RANDOM_STATE, CV_FOLDS


class F1ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def split_data(self, X, y):
        return train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
    
    def train_random_forest(self, X_train, y_train):
        model = RandomForestRegressor(
            n_estimators=MODEL_PARAMS["random_forest"]["n_estimators"],
            max_depth=MODEL_PARAMS["random_forest"]["max_depth"],
            random_state=MODEL_PARAMS["random_forest"]["random_state"]
        )
        model.fit(X_train, y_train)
        return model
    
    def train_gradient_boosting(self, X_train, y_train):
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)
        return model
    
    def train_linear_regression(self, X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test, X_full=None, y_full=None):
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        cv_mae = None
        if X_full is not None and y_full is not None:
            cv_scores = cross_val_score(
                model, X_full, y_full,
                cv=CV_FOLDS,
                scoring="neg_mean_absolute_error"
            )
            cv_mae = -cv_scores.mean()
        
        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "cv_mae": cv_mae,
            "y_pred": y_pred
        }
    
    def train_all_models(self, X, y):
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        print("Training Random Forest...")
        rf_model = self.train_random_forest(X_train, y_train)
        self.models["random_forest"] = rf_model
        self.results["random_forest"] = self.evaluate_model(
            rf_model, X_test, y_test, X, y
        )
        
        print("Training Gradient Boosting...")
        gb_model = self.train_gradient_boosting(X_train, y_train)
        self.models["gradient_boosting"] = gb_model
        self.results["gradient_boosting"] = self.evaluate_model(
            gb_model, X_test, y_test, X, y
        )
        
        print("Training Linear Regression...")
        lr_model = self.train_linear_regression(X_train, y_train)
        self.models["linear_regression"] = lr_model
        self.results["linear_regression"] = self.evaluate_model(
            lr_model, X_test, y_test, X, y
        )
        
        best_r2 = -float("inf")
        for name, result in self.results.items():
            if result["r2"] > best_r2:
                best_r2 = result["r2"]
                self.best_model_name = name
                self.best_model = self.models[name]
        
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }
    
    def get_model_comparison(self):
        comparison = []
        for name, result in self.results.items():
            comparison.append({
                "Model": name,
                "MAE": result["mae"],
                "RMSE": result["rmse"],
                "R2": result["r2"],
                "CV_MAE": result["cv_mae"]
            })
        return pd.DataFrame(comparison)
    
    def get_feature_importance(self, model_name="random_forest"):
        model = self.models.get(model_name)
        if model is not None and hasattr(model, "feature_importances_"):
            return model.feature_importances_
        return None
    
    def predict(self, X, model_name=None):
        if model_name is None:
            model = self.best_model
        else:
            model = self.models.get(model_name)
        
        if model is not None:
            return model.predict(X)
        return None
    
    def get_results(self):
        return self.results
    
    def get_models(self):
        return self.models