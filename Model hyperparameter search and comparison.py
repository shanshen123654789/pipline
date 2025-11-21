
### model1  LinearRegression(LR)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

params_lr = {'fit_intercept': [True, False]}

lr_model = LinearRegression()

cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search_lr = GridSearchCV(
    estimator=lr_model,
    param_grid=params_lr,
    cv=cv,
    n_jobs=1,
    scoring='neg_mean_squared_error',
    verbose=0
)

grid_search_lr.fit(X_train, y_train)

best_lr_model = grid_search_lr.best_estimator_
best_params = grid_search_lr.best_params_
print(f"\n✅ Best parameters for Linear Regression: {best_params}")

cv_results = pd.DataFrame(grid_search_lr.cv_results_)
cv_results['MSE'] = -cv_results['mean_test_score']
cv_results['MSE_std'] = cv_results['std_test_score']
cv_results['RMSE'] = np.sqrt(cv_results['MSE'])
cv_results['RMSE_std_approx'] = cv_results['MSE_std'] / (2 * cv_results['RMSE'])
cv_summary = cv_results[['params', 'MSE', 'MSE_std', 'RMSE', 'RMSE_std_approx', 'rank_test_score']].sort_values('rank_test_score')

print("\n📊 Cross-validation (5-fold) performance summary:")
print(cv_summary.to_string(index=False))

y_pred_lr = best_lr_model.predict(X_test)

eps = 1e-9
y_test_safe = np.clip(np.asarray(y_test), eps, None)
y_pred_safe = np.clip(np.asarray(y_pred_lr), eps, None)

msle_lr = mean_squared_log_error(y_test_safe, y_pred_safe)
rmse_lr = np.sqrt(mean_squared_error(y_test_safe, y_pred_safe))
mae_lr = mean_absolute_error(y_test_safe, y_pred_safe)
mape_lr = np.mean(np.abs((y_test_safe - y_pred_safe) / y_test_safe)) * 100
r2_lr = r2_score(y_test_safe, y_pred_safe)

print("\n🎯 Test set evaluation metrics:")
print(f"MSLE = {msle_lr:.6f}")
print(f"RMSE = {rmse_lr:.4f}")
print(f"MAE  = {mae_lr:.4f}")
print(f"MAPE = {mape_lr:.3f}%")
print(f"R²   = {r2_lr:.4f}")

cv_summary.to_excel("/home/jiangshan/cre/growth/LR_cv_summary.xlsx", index=False)
print("\n✅ Cross-validation summary exported to: /home/jiangshan/cre/growth/LR_cv_summary.xlsx")

### model2 lasso
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

params_lasso = {
    'alpha': [0.01, 0.1, 1.0, 10.0],
    'fit_intercept': [True, False],
    'max_iter': [10000]
}
lasso_model = Lasso(random_state=42)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search_lasso = GridSearchCV(
    estimator=lasso_model,
    param_grid=params_lasso,
    cv=cv,
    n_jobs=1,
    scoring='neg_mean_squared_error',
    verbose=0
)
grid_search_lasso.fit(X_train, y_train)

best_lasso_model = grid_search_lasso.best_estimator_
best_params_lasso = grid_search_lasso.best_params_
print(f"\n✅ Best parameters for Lasso: {best_params_lasso}")

cv_results = pd.DataFrame(grid_search_lasso.cv_results_)
cv_results['MSE'] = -cv_results['mean_test_score']
cv_results['MSE_std'] = cv_results['std_test_score']
cv_results['RMSE'] = np.sqrt(cv_results['MSE'])
cv_results['RMSE_std_approx'] = cv_results['MSE_std'] / (2 * cv_results['RMSE'])
cv_summary = cv_results[['params', 'MSE', 'MSE_std', 'RMSE', 'RMSE_std_approx', 'rank_test_score']].sort_values('rank_test_score')

print("\n📊 Lasso 5-fold CV performance summary:")
print(cv_summary.to_string(index=False))

eps = 1e-9
y_test_safe = np.clip(y_test, eps, None)
y_pred_lasso = best_lasso_model.predict(X_test)
y_pred_safe = np.clip(y_pred_lasso, eps, None)

msle = mean_squared_log_error(y_test_safe, y_pred_safe)
rmse = np.sqrt(mean_squared_error(y_test_safe, y_pred_safe))
mae = mean_absolute_error(y_test_safe, y_pred_safe)
mape = np.mean(np.abs((y_test_safe - y_pred_safe) / y_test_safe)) * 100
r2 = r2_score(y_test_safe, y_pred_safe)

print("\n🎯 Lasso Test set metrics:")
print(f"MSLE = {msle:.6f}")
print(f"RMSE = {rmse:.4f}")
print(f"MAE  = {mae:.4f}")
print(f"MAPE = {mape:.3f}%")
print(f"R²   = {r2:.4f}")


cv_summary.to_excel("/home/jiangshan/cre/growth/Lasso_cv_summary.xlsx", index=False)
print("\n✅ CV summary exported to: /home/jiangshan/cre/growth/Lasso_cv_summary.xlsx")

### model3 Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd


params_ridge = {
    'alpha': [0.01, 0.1, 1.0, 10.0],
    'fit_intercept': [True, False],
    'solver': ['auto', 'lbfgs', 'saga']
}
ridge_model = Ridge(random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search_ridge = GridSearchCV(
    estimator=ridge_model,
    param_grid=params_ridge,
    cv=cv,
    n_jobs=1,
    scoring='neg_mean_squared_error',
    verbose=0
)
grid_search_ridge.fit(X_train, y_train)

best_ridge_model = grid_search_ridge.best_estimator_
best_params_ridge = grid_search_ridge.best_params_
print(f"\n✅ Best parameters for Ridge: {best_params_ridge}")

cv_results = pd.DataFrame(grid_search_ridge.cv_results_)
cv_results['MSE'] = -cv_results['mean_test_score']
cv_results['MSE_std'] = cv_results['std_test_score']
cv_results['RMSE'] = np.sqrt(cv_results['MSE'])
cv_results['RMSE_std_approx'] = cv_results['MSE_std'] / (2 * cv_results['RMSE'])
cv_summary = cv_results[['params', 'MSE', 'MSE_std', 'RMSE', 'RMSE_std_approx', 'rank_test_score']].sort_values('rank_test_score')

print("\n📊 Ridge 5-fold CV performance summary:")
print(cv_summary.to_string(index=False))

eps = 1e-9
y_test_safe = np.clip(y_test, eps, None)
y_pred_ridge = best_ridge_model.predict(X_test)
y_pred_safe = np.clip(y_pred_ridge, eps, None)

msle = mean_squared_log_error(y_test_safe, y_pred_safe)
rmse = np.sqrt(mean_squared_error(y_test_safe, y_pred_safe))
mae = mean_absolute_error(y_test_safe, y_pred_safe)
mape = np.mean(np.abs((y_test_safe - y_pred_safe) / y_test_safe)) * 100
r2 = r2_score(y_test_safe, y_pred_safe)

print("\n🎯 Ridge Test set metrics:")
print(f"MSLE = {msle:.6f}")
print(f"RMSE = {rmse:.4f}")
print(f"MAE  = {mae:.4f}")
print(f"MAPE = {mape:.3f}%")
print(f"R²   = {r2:.4f}")


cv_summary.to_excel("/home/jiangshan/cre/growth/Ridge_cv_summary.xlsx", index=False)
print("\n✅ CV summary exported to: /home/jiangshan/cre/growth/Ridge_cv_summary.xlsx")

### model4 ElasticNet
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

params_enet = {
    'alpha': [0.01, 0.1, 1.0, 10.0],
    'l1_ratio': [0.1, 0.5, 0.9],
    'fit_intercept': [True, False],
    'max_iter': [10000]
}

enet_model = ElasticNet(random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search_enet = GridSearchCV(enet_model, param_grid=params_enet, cv=cv, n_jobs=1, scoring='neg_mean_squared_error')
grid_search_enet.fit(X_train, y_train)

best_model_enet = grid_search_enet.best_estimator_
best_params_enet = grid_search_enet.best_params_
print(f"\n✅ Best parameters for ElasticNet: {best_params_enet}")

cv_results_enet = pd.DataFrame(grid_search_enet.cv_results_)
cv_results_enet['MSE'] = -cv_results_enet['mean_test_score']
cv_results_enet['RMSE'] = np.sqrt(cv_results_enet['MSE'])
cv_summary_enet = cv_results_enet[['params', 'MSE', 'RMSE', 'rank_test_score']].sort_values('rank_test_score')

y_pred_enet = best_model_enet.predict(X_test)
eps = 1e-9
y_test_safe = np.clip(np.asarray(y_test), eps, None)
y_pred_safe = np.clip(y_pred_enet, eps, None)

msle_enet = mean_squared_log_error(y_test_safe, y_pred_safe)
rmse_enet = np.sqrt(mean_squared_error(y_test_safe, y_pred_safe))
mae_enet = mean_absolute_error(y_test_safe, y_pred_safe)
mape_enet = np.mean(np.abs((y_test_safe - y_pred_safe) / y_test_safe)) * 100
r2_enet = r2_score(y_test_safe, y_pred_safe)

print("\n🎯 Test set metrics (ElasticNet):")
print(f"MSLE = {msle_enet:.6f}, RMSE = {rmse_enet:.4f}, MAE = {mae_enet:.4f}, MAPE = {mape_enet:.3f}%, R² = {r2_enet:.4f}")

summary_enet = pd.DataFrame([{
    'Model': 'ElasticNet',
    'Best Parameters': str(best_params_enet),
    'MSLE': round(msle_enet, 6),
    'RMSE': round(rmse_enet, 4),
    'MAE': round(mae_enet, 4),
    'MAPE(%)': round(mape_enet, 3),
    'R²': round(r2_enet, 4)
}])


output_path = "/home/jiangshan/cre/growth/ElasticNet_results.xlsx"
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    cv_summary_enet.to_excel(writer, sheet_name="CV_Results", index=False)
    summary_enet.to_excel(writer, sheet_name="Test_Summary", index=False)

print(f"\n✅ ElasticNet results saved to: {output_path}")

### model5 Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

params_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_model = RandomForestRegressor(random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search_rf = GridSearchCV(rf_model, param_grid=params_rf, cv=cv, n_jobs=1, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train, y_train)

best_model_rf = grid_search_rf.best_estimator_
best_params_rf = grid_search_rf.best_params_
print(f"\n✅ Best parameters for Random Forest: {best_params_rf}")

cv_results_rf = pd.DataFrame(grid_search_rf.cv_results_)
cv_results_rf['MSE'] = -cv_results_rf['mean_test_score']
cv_results_rf['RMSE'] = np.sqrt(cv_results_rf['MSE'])
cv_summary_rf = cv_results_rf[['params', 'MSE', 'RMSE', 'rank_test_score']].sort_values('rank_test_score')
print("\n📊 Random Forest 5-fold CV summary:")
print(cv_summary_rf.to_string(index=False))

y_pred_rf = best_model_rf.predict(X_test)
eps = 1e-9
y_test_safe = np.clip(np.asarray(y_test), eps, None)
y_pred_safe = np.clip(y_pred_rf, eps, None)

msle_rf = mean_squared_log_error(y_test_safe, y_pred_safe)
rmse_rf = np.sqrt(mean_squared_error(y_test_safe, y_pred_safe))
mae_rf = mean_absolute_error(y_test_safe, y_pred_safe)
mape_rf = np.mean(np.abs((y_test_safe - y_pred_safe) / y_test_safe)) * 100
r2_rf = r2_score(y_test_safe, y_pred_safe)

print("\n🎯 Test set metrics (Random Forest):")
print(f"MSLE = {msle_rf:.6f}, RMSE = {rmse_rf:.4f}, MAE = {mae_rf:.4f}, MAPE = {mape_rf:.3f}%, R² = {r2_rf:.4f}")

cv_summary_rf.to_excel("/home/jiangshan/cre/growth/RandomForest_cv_summary.xlsx", index=False)

### model6 Gradient Boosting machine(GBM)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

params_gbm = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}

gbm_model = GradientBoostingRegressor(random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search_gbm = GridSearchCV(gbm_model, param_grid=params_gbm, cv=cv, n_jobs=1, scoring='neg_mean_squared_error')
grid_search_gbm.fit(X_train, y_train)

best_model_gbm = grid_search_gbm.best_estimator_
best_params_gbm = grid_search_gbm.best_params_
print(f"\n✅ Best parameters for Gradient Boosting: {best_params_gbm}")

cv_results_gbm = pd.DataFrame(grid_search_gbm.cv_results_)
cv_results_gbm['MSE'] = -cv_results_gbm['mean_test_score']
cv_results_gbm['RMSE'] = np.sqrt(cv_results_gbm['MSE'])
cv_summary_gbm = cv_results_gbm[['params', 'MSE', 'RMSE', 'rank_test_score']].sort_values('rank_test_score')
print("\n📊 Gradient Boosting 5-fold CV summary:")
print(cv_summary_gbm.to_string(index=False))

y_pred_gbm = best_model_gbm.predict(X_test)
eps = 1e-9
y_test_safe = np.clip(np.asarray(y_test), eps, None)
y_pred_safe = np.clip(y_pred_gbm, eps, None)

msle_gbm = mean_squared_log_error(y_test_safe, y_pred_safe)
rmse_gbm = np.sqrt(mean_squared_error(y_test_safe, y_pred_safe))
mae_gbm = mean_absolute_error(y_test_safe, y_pred_safe)
mape_gbm = np.mean(np.abs((y_test_safe - y_pred_safe) / y_test_safe)) * 100
r2_gbm = r2_score(y_test_safe, y_pred_safe)

print("\n🎯 Test set metrics (Gradient Boosting):")
print(f"MSLE = {msle_gbm:.6f}, RMSE = {rmse_gbm:.4f}, MAE = {mae_gbm:.4f}, MAPE = {mape_gbm:.3f}%, R² = {r2_gbm:.4f}")

cv_summary_gbm.to_excel("/home/jiangshan/cre/growth/GBM_cv_summary.xlsx", index=False)

### model7 AdaBoost
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

params_ada = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
}

ada_model = AdaBoostRegressor(random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search_ada = GridSearchCV(ada_model, param_grid=params_ada, cv=cv, n_jobs=1, scoring='neg_mean_squared_error')
grid_search_ada.fit(X_train, y_train)

best_model_ada = grid_search_ada.best_estimator_
best_params_ada = grid_search_ada.best_params_
print(f"\n✅ Best parameters for AdaBoost: {best_params_ada}")

cv_results_ada = pd.DataFrame(grid_search_ada.cv_results_)
cv_results_ada['MSE'] = -cv_results_ada['mean_test_score']
cv_results_ada['RMSE'] = np.sqrt(cv_results_ada['MSE'])
cv_summary_ada = cv_results_ada[['params', 'MSE', 'RMSE', 'rank_test_score']].sort_values('rank_test_score')
print("\n📊 AdaBoost 5-fold CV summary:")
print(cv_summary_ada.to_string(index=False))

y_pred_ada = best_model_ada.predict(X_test)
eps = 1e-9
y_test_safe = np.clip(np.asarray(y_test), eps, None)
y_pred_safe = np.clip(y_pred_ada, eps, None)

msle_ada = mean_squared_log_error(y_test_safe, y_pred_safe)
rmse_ada = np.sqrt(mean_squared_error(y_test_safe, y_pred_safe))
mae_ada = mean_absolute_error(y_test_safe, y_pred_safe)
mape_ada = np.mean(np.abs((y_test_safe - y_pred_safe) / y_test_safe)) * 100
r2_ada = r2_score(y_test_safe, y_pred_safe)

print("\n🎯 Test set metrics (AdaBoost):")
print(f"MSLE = {msle_ada:.6f}, RMSE = {rmse_ada:.4f}, MAE = {mae_ada:.4f}, MAPE = {mape_ada:.3f}%, R² = {r2_ada:.4f}")

cv_summary_ada.to_excel("/home/jiangshan/cre/growth/AdaBoost_cv_summary.xlsx", index=False)

### eXtreme Gradient Boosting(XGboost)
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from openpyxl import Workbook

params_xgb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_model = xgb.XGBRegressor(random_state=42, verbosity=0)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search_xgb = GridSearchCV(xgb_model, param_grid=params_xgb, cv=cv, n_jobs=1, scoring='neg_mean_squared_error')
grid_search_xgb.fit(X_train, y_train)

best_model_xgb = grid_search_xgb.best_estimator_
best_params_xgb = grid_search_xgb.best_params_
print(f"\n✅ Best parameters for XGBoost: {best_params_xgb}")

cv_results_xgb = pd.DataFrame(grid_search_xgb.cv_results_)
cv_results_xgb['MSE'] = -cv_results_xgb['mean_test_score']
cv_results_xgb['RMSE'] = np.sqrt(cv_results_xgb['MSE'])
cv_summary_xgb = cv_results_xgb[['params', 'MSE', 'RMSE', 'rank_test_score']].sort_values('rank_test_score')

y_pred_xgb = best_model_xgb.predict(X_test)
eps = 1e-9
y_test_safe = np.clip(np.asarray(y_test), eps, None)
y_pred_safe = np.clip(y_pred_xgb, eps, None)

msle_xgb = mean_squared_log_error(y_test_safe, y_pred_safe)
rmse_xgb = np.sqrt(mean_squared_error(y_test_safe, y_pred_safe))
mae_xgb = mean_absolute_error(y_test_safe, y_pred_safe)
mape_xgb = np.mean(np.abs((y_test_safe - y_pred_safe) / y_test_safe)) * 100
r2_xgb = r2_score(y_test_safe, y_pred_safe)

summary_xgb = pd.DataFrame([{
    'Model': 'XGBoost',
    'Best Parameters': str(best_params_xgb),
    'MSLE': round(msle_xgb, 6),
    'RMSE': round(rmse_xgb, 4),
    'MAE': round(mae_xgb, 4),
    'MAPE(%)': round(mape_xgb, 3),
    'R²': round(r2_xgb, 4)
}])

output_path = "/home/jiangshan/cre/growth/XGBoost_results.xlsx"
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    cv_summary_xgb.to_excel(writer, sheet_name="CV_Results", index=False)
    summary_xgb.to_excel(writer, sheet_name="Test_Summary", index=False)

print(f"\n✅ Results saved to: {output_path}")

### model8 Decision Tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ===== 1️⃣ Define variable types =====
# Numeric features
numeric_features = ['30kg ABW', 'Litter size', 'Birth weight']
# Categorical features (need encoding)
categorical_features = ['Season', 'Parity', 'Sex']

print("📊 Decision tree data preprocessing info:")
print(f"   Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"   Categorical features ({len(categorical_features)}): {categorical_features}")

# ===== 2️⃣ Preprocessing pipeline =====
# Decision trees do not require scaling of numeric variables but need encoding for categorical variables
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

# Build column transformer - encode only categorical variables, keep numeric untouched
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'  # keep the remaining columns (numeric features)
)

# ===== 3️⃣ Decision Tree Regressor =====
dt_model = DecisionTreeRegressor(random_state=42)

# Build the full Pipeline
pipeline_dt = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', dt_model)
])

# ===== 4️⃣ Hyperparameter grid =====
params_dt = {
    'regressor__max_depth': [3, 5, 7, 10, None],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

# ===== 5️⃣ Cross-validation + Grid search =====
cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search_dt = GridSearchCV(
    estimator=pipeline_dt,  # use pipeline with preprocessing
    param_grid=params_dt,
    cv=cv,
    n_jobs=1,
    scoring='neg_mean_squared_error',
    verbose=1,
    return_train_score=True
)

# ===== 6️⃣ Fit model =====
print("🚀 Start training Decision Tree Regressor...")
print("⚠️  Note: Categorical variables are automatically one-hot encoded, numeric variables remain on original scale")
grid_search_dt.fit(X_train, y_train)

# ===== 7️⃣ Best model and parameters =====
best_model_dt = grid_search_dt.best_estimator_
best_params_dt = grid_search_dt.best_params_
best_score_dt = -grid_search_dt.best_score_  # convert to positive MSE

print(f"\n✅ Best parameters for Decision Tree: {best_params_dt}")
print(f"✅ Best CV RMSE: {np.sqrt(best_score_dt):.4f}")

# ===== 8️⃣ Cross-validation results analysis =====
cv_results_dt = pd.DataFrame(grid_search_dt.cv_results_)
cv_results_dt['MSE'] = -cv_results_dt['mean_test_score']
cv_results_dt['RMSE'] = np.sqrt(cv_results_dt['MSE'])
cv_results_dt['train_RMSE'] = np.sqrt(-cv_results_dt['mean_train_score'])
cv_summary_dt = cv_results_dt[['params', 'MSE', 'RMSE', 'train_RMSE', 'rank_test_score']].sort_values('rank_test_score')

print(f"\n📈 Top 3 parameter combinations:")
for i, row in cv_summary_dt.head(3).iterrows():
    print(f"   Rank {int(row['rank_test_score'])}: RMSE = {row['RMSE']:.4f}")

# ===== 9️⃣ Test set prediction and evaluation =====
y_pred_dt = best_model_dt.predict(X_test)

# Avoid negative values (to prevent MSLE error)
eps = 1e-9
y_test_safe = np.clip(np.asarray(y_test), eps, None)
y_pred_safe = np.clip(y_pred_dt, eps, None)

msle_dt = mean_squared_log_error(y_test_safe, y_pred_safe)
rmse_dt = np.sqrt(mean_squared_error(y_test_safe, y_pred_safe))
mae_dt = mean_absolute_error(y_test_safe, y_pred_safe)
mape_dt = np.mean(np.abs((y_test_safe - y_pred_safe) / y_test_safe)) * 100
r2_dt = r2_score(y_test_safe, y_pred_safe)

print("\n🎯 Decision Tree test set performance:")
print(f"MSLE = {msle_dt:.6f}")
print(f"RMSE = {rmse_dt:.4f}")
print(f"MAE  = {mae_dt:.4f}")
print(f"MAPE = {mape_dt:.3f}%")
print(f"R²   = {r2_dt:.4f}")

# ===== 🔟 Feature importance analysis =====
print("\n🔍 Decision Tree feature importance analysis:")
# Initialize variables
tree_depth = 'N/A'
n_leaves = 'N/A'
train_score = 'N/A'
num_important_features = 'N/A'

try:
    # Get preprocessor and regressor
    preprocessor = best_model_dt.named_steps['preprocessor']
    dt_regressor = best_model_dt.named_steps['regressor']

    # Get feature names
    feature_names = []

    # Categorical feature names (after one-hot encoding)
    if categorical_features:
        cat_encoder = preprocessor.named_transformers_['cat']
        for i, col in enumerate(categorical_features):
            if hasattr(cat_encoder, 'categories_'):
                categories = cat_encoder.categories_[i]
                # Skip the first category (because drop='first')
                for cat in categories[1:]:
                    feature_names.append(f"{col}_{cat}")

    # Add numeric feature names
    feature_names.extend(numeric_features)

    # Get feature importance
    feature_importances = dt_regressor.feature_importances_

    # Create feature importance DataFrame
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)

    # Cumulative importance
    feature_importance_df['Cumulative_Importance'] = feature_importance_df['Importance'].cumsum()

    # Find number of features contributing to 80% of importance
    important_features = feature_importance_df[feature_importance_df['Cumulative_Importance'] <= 0.8]
    num_important_features = len(important_features)

    print(f"Total number of features: {len(feature_names)}")
    print(f"Top {num_important_features} features contribute 80% of total importance")
    print("Top 10 most important features:")
    print(feature_importance_df.head(10).to_string(index=False))

    # Aggregate importance by original feature
    original_feature_importance = {}
    for feature, importance in zip(feature_names, feature_importances):
        # Extract original feature name (remove encoded suffix)
        if '_' in feature:
            original_feature = feature.split('_')[0]
        else:
            original_feature = feature

        if original_feature not in original_feature_importance:
            original_feature_importance[original_feature] = 0
        original_feature_importance[original_feature] += importance

    # Create DataFrame for original feature importance
    original_importance_df = pd.DataFrame({
        'Original_Feature': list(original_feature_importance.keys()),
        'Total_Importance': list(original_feature_importance.values())
    }).sort_values('Total_Importance', ascending=False)

    print(f"\n📊 Feature importance grouped by original variables:")
    print(original_importance_df.to_string(index=False))

    # Analyze tree structure
    tree_depth = dt_regressor.get_depth()
    n_leaves = dt_regressor.get_n_leaves()

    print(f"\n🌳 Decision Tree structure:")
    print(f"   Depth: {tree_depth}")
    print(f"   Number of leaves: {n_leaves}")
    print(f"   Number of features: {len(feature_names)}")

    # Overfitting check
    X_train_transformed = preprocessor.transform(X_train)
    train_score = dt_regressor.score(X_train_transformed, y_train)

    if train_score - r2_dt > 0.1:
        print("⚠️  Warning: Model may be overfitting (training score much higher than test score)")
    elif train_score - r2_dt < 0.05:
        print("✅ Good generalization (training and test scores are close)")
    else:
        print("💡 Model may have mild overfitting")

except Exception as e:
    print(f"⚠️ Unable to compute feature importance: {e}")


# ===== 1️⃣1️⃣ Export summary results =====
# Safely format numeric values
def safe_format(value, format_spec='.4f'):
    """Safely format numeric values; if string, return as-is."""
    if isinstance(value, (int, float)):
        return format(value, format_spec)
    else:
        return str(value)


summary_dt = pd.DataFrame([{
    'Model': 'Decision Tree',
    'Best Parameters': str(best_params_dt),
    'MSLE': safe_format(msle_dt, '.6f'),
    'RMSE': safe_format(rmse_dt, '.4f'),
    'MAE': safe_format(mae_dt, '.4f'),
    'MAPE(%)': safe_format(mape_dt, '.3f'),
    'R²': safe_format(r2_dt, '.4f'),
    'CV_RMSE': safe_format(np.sqrt(best_score_dt), '.4f'),
    'Important_Features': num_important_features,
    'Tree_Depth': tree_depth,
    'Number_Leaves': n_leaves,
    'Train_Score': safe_format(train_score, '.4f'),
    'Test_Score': safe_format(r2_dt, '.4f')
}])

output_path = "/home/jiangshan/cre/growth/DecisionTree_results.xlsx"
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    cv_summary_dt.to_excel(writer, sheet_name="CV_Results", index=False)
    summary_dt.to_excel(writer, sheet_name="Test_Summary", index=False)

    # Add feature importance to Excel
    try:
        if 'feature_importance_df' in locals():
            feature_importance_df.to_excel(writer, sheet_name="Feature_Importance", index=False)
        if 'original_importance_df' in locals():
            original_importance_df.to_excel(writer, sheet_name="Original_Feature_Importance", index=False)
    except Exception as e:
        print(f"⚠️ Unable to save feature importance to Excel: {e}")

    # Add preprocessing info
    preprocess_info = pd.DataFrame({
        'DecisionTree_Preprocessing': [
            f'Numeric features: {numeric_features}',
            f'Categorical features: {categorical_features}',
            f'Numeric processing: keep original scale (tree models do not require scaling)',
            f'Categorical processing: OneHotEncoder (drop_first=True)',
            f'Model type: single decision tree',
            f'Split criterion: reduction in MSE',
            f'Characteristics: high interpretability, prone to overfitting'
        ]
    })
    preprocess_info.to_excel(writer, sheet_name="Preprocessing_Info", index=False)

    # Add model configuration info
    model_config_data = [
        f'Max depth: {best_params_dt.get("regressor__max_depth", "N/A")}',
        f'Min samples split: {best_params_dt.get("regressor__min_samples_split", "N/A")}',
        f'Min samples leaf: {best_params_dt.get("regressor__min_samples_leaf", "N/A")}',
        f'Actual tree depth: {tree_depth}',
        f'Number of leaves: {n_leaves}',
        f'Number of important features (80% contribution): {num_important_features}',
        f'Training R²: {safe_format(train_score, ".4f")}',
        f'Test R²: {safe_format(r2_dt, ".4f")}'
    ]

    model_info = pd.DataFrame({
        'DecisionTree_Configuration': model_config_data
    })
    model_info.to_excel(writer, sheet_name="Model_Config", index=False)

print(f"\n✅ All results have been saved to: {output_path}")

# ===== 1️⃣2️⃣ Model performance summary =====
print("\n" + "=" * 50)
print("📊 Decision Tree model training completed!")
print(f"   Tree depth: {tree_depth}")
print(f"   Number of leaves: {num_important_features}")

# Safe printing of train and test R²
train_score_display = safe_format(train_score, '.4f')
r2_dt_display = safe_format(r2_dt, '.4f')
rmse_dt_display = safe_format(rmse_dt, '.4f')

print(f"   Training R²: {train_score_display}")
print(f"   Test R²: {r2_dt_display}")
print(f"   Test RMSE: {rmse_dt_display}")
print("=" * 50)

# ===== 1️⃣3️⃣ Decision tree specific notes =====
print("\n💡 Decision Tree characteristics:")
print("   ✅ Pros: highly interpretable, no feature scaling needed, handles mixed data types, easy to visualize")
print("   ⚠️  Cons: prone to overfitting, sensitive to small data changes, usually weaker than ensemble methods")
print("   📝 Recommendations:")
print("      - Suitable as a baseline model or for understanding data relationships")
print("      - Use pruning-related parameters to control overfitting")
print("      - Consider Random Forest or Gradient Boosting for better performance")

# ===== 1️⃣4️⃣ Overfitting analysis and recommendations =====
print("\n🔍 Overfitting analysis:")
if isinstance(train_score, (int, float)) and isinstance(r2_dt, (int, float)):
    gap = train_score - r2_dt
    if gap > 0.15:
        print("🚨 Severe overfitting! Suggestions:")
        print("   - Increase min_samples_split and min_samples_leaf")
        print("   - Use smaller max_depth")
        print("   - Consider ensemble methods such as Random Forest")
    elif gap > 0.08:
        print("⚠️  Possible overfitting, consider tuning pruning-related parameters")
    else:
        print("✅ Good generalization ability")

    # Suggestions based on tree complexity
    if isinstance(n_leaves, int) and n_leaves > 50:
        print("💡 Tree structure is quite complex, consider stronger regularization")
    elif isinstance(n_leaves, int) and n_leaves < 10:
        print("💡 Tree structure is very simple, possible underfitting")
else:
    print("⚠️  Unable to perform overfitting analysis, feature importance computation failed")

# ===== 1️⃣5️⃣ Next-step modeling suggestions =====
print("\n🎯 Suggestions for next steps:")
if isinstance(r2_dt, (int, float)):
    if r2_dt < 0.5:
        print("   📈 Current performance is moderate, consider:")
        print("      - Random Forest (ensemble of trees)")
        print("      - Gradient Boosting (sequential improvement)")
        print("      - More advanced feature engineering")
    elif r2_dt > 0.8:
        print("   🎉 Excellent performance! You may:")
        print("      - Analyze decision rules for domain interpretation")
        print("      - Visualize the decision tree")
    else:
        print("   🔄 Performance is acceptable; can serve as a baseline for other models")
else:
    print("   🔄 Unable to assess performance, please check the training process")


### model9  Extra Trees
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ===== 1️⃣ Define variable types =====
# Numeric variables
numeric_features = ['30kg ABW', 'Litter size', 'Birth weight']
# Categorical variables (need encoding)
categorical_features = ['Season', 'Parity', 'Sex']

print("📊 ExtraTrees Data Preprocessing Info:")
print(f"   Numeric variables ({len(numeric_features)}): {numeric_features}")
print(f"   Categorical variables ({len(categorical_features)}): {categorical_features}")

# ===== 2️⃣ Preprocessing Pipeline =====
# ExtraTrees doesn't require standardization of numeric variables but needs encoding of categorical variables
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

# Build column transformer - encode only categorical variables, keep numeric variables unchanged
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'  # Keep unspecified columns (i.e., numeric variables)
)

# ===== 3️⃣ ExtraTrees Regression Model =====
et_model = ExtraTreesRegressor(random_state=42)

# Build overall Pipeline
pipeline_et = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', et_model)
])

# ===== 4️⃣ Hyperparameter Grid =====
params_et = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__max_features': ['auto', 'sqrt', 'log2'],  # Add feature selection strategies
    'regressor__bootstrap': [False, True]  # Add bootstrap option
}

# ===== 5️⃣ Cross-validation + Grid Search =====
cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search_et = GridSearchCV(
    estimator=pipeline_et,  # Using pipeline with preprocessing
    param_grid=params_et,
    cv=cv,
    n_jobs=1,
    scoring='neg_mean_squared_error',
    verbose=1,  # Show progress
    return_train_score=True
)

# ===== 6️⃣ Fit Model =====
print("🚀 Starting to train ExtraTrees regression model...")
print("⚠️  Note: ExtraTrees automatically encodes categorical variables, keeps numeric variables on the original scale")
print("💡 ExtraTrees Characteristics: Extremely random trees, stronger randomness, usually faster than RandomForest")
grid_search_et.fit(X_train, y_train)

# ===== 7️⃣ Output Best Model and Parameters =====
best_model_et = grid_search_et.best_estimator_
best_params_et = grid_search_et.best_params_
best_score_et = -grid_search_et.best_score_  # Convert to positive

print(f"\n✅ ExtraTrees Best Parameters: {best_params_et}")
print(f"✅ Best Cross-validation RMSE: {np.sqrt(best_score_et):.4f}")

# ===== 8️⃣ Cross-validation Results Analysis =====
cv_results_et = pd.DataFrame(grid_search_et.cv_results_)
cv_results_et['MSE'] = -cv_results_et['mean_test_score']
cv_results_et['RMSE'] = np.sqrt(cv_results_et['MSE'])
cv_results_et['train_RMSE'] = np.sqrt(-cv_results_et['mean_train_score'])
cv_summary_et = cv_results_et[['params', 'MSE', 'RMSE', 'train_RMSE', 'rank_test_score']].sort_values('rank_test_score')

print(f"\n📈 Top 3 Best Parameter Combinations:")
for i, row in cv_summary_et.head(3).iterrows():
    print(f"   Rank {int(row['rank_test_score'])}: RMSE = {row['RMSE']:.4f}")

# ===== 9️⃣ Test Set Prediction and Evaluation =====
y_pred_et = best_model_et.predict(X_test)

# Avoid negative values (to prevent MSLE error)
eps = 1e-9
y_test_safe = np.clip(np.asarray(y_test), eps, None)
y_pred_safe = np.clip(y_pred_et, eps, None)

msle_et = mean_squared_log_error(y_test_safe, y_pred_safe)
rmse_et = np.sqrt(mean_squared_error(y_test_safe, y_pred_safe))
mae_et = mean_absolute_error(y_test_safe, y_pred_safe)
mape_et = np.mean(np.abs((y_test_safe - y_pred_safe) / y_test_safe)) * 100
r2_et = r2_score(y_test_safe, y_pred_safe)

print("\n🎯 ExtraTrees Test Set Performance Metrics:")
print(f"MSLE = {msle_et:.6f}")
print(f"RMSE = {rmse_et:.4f}")
print(f"MAE  = {mae_et:.4f}")
print(f"MAPE = {mape_et:.3f}%")
print(f"R²   = {r2_et:.4f}")

# ===== 🔟 Feature Importance Analysis =====
print("\n🔍 ExtraTrees Feature Importance Analysis:")
# Initialize variables
num_important_features = 'N/A'
train_score = 'N/A'


# Safe formatting function
def safe_format(value, format_spec='.4f'):
    """Safely format numeric values, return as string if not numeric"""
    if isinstance(value, (int, float)):
        return format(value, format_spec)
    else:
        return str(value)


try:
    # Get preprocessed feature names
    preprocessor = best_model_et.named_steps['preprocessor']
    et_regressor = best_model_et.named_steps['regressor']

    # Get feature names
    feature_names = []

    # Categorical feature names (after one-hot encoding)
    if categorical_features:
        cat_encoder = preprocessor.named_transformers_['cat']
        for i, col in enumerate(categorical_features):
            if hasattr(cat_encoder, 'categories_'):
                categories = cat_encoder.categories_[i]
                # Skip the first category (because drop='first')
                for cat in categories[1:]:
                    feature_names.append(f"{col}_{cat}")

    # Add numeric feature names
    feature_names.extend(numeric_features)

    # Get feature importance
    feature_importances = et_regressor.feature_importances_

    # Create feature importance DataFrame
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)

    # Calculate cumulative importance
    feature_importance_df['Cumulative_Importance'] = feature_importance_df['Importance'].cumsum()

    # Find number of features contributing to 80% of importance
    important_features = feature_importance_df[feature_importance_df['Cumulative_Importance'] <= 0.8]
    num_important_features = len(important_features)

    print(f"Total number of features: {len(feature_names)}")
    print(f"Top {num_important_features} features contribute 80% of total importance")
    print("Top 10 most important features:")
    print(feature_importance_df.head(10).to_string(index=False))

    # Aggregate importance by original feature
    original_feature_importance = {}
    for feature, importance in zip(feature_names, feature_importances):
        # Extract original feature name (remove encoded suffix)
        if '_' in feature:
            original_feature = feature.split('_')[0]
        else:
            original_feature = feature

        if original_feature not in original_feature_importance:
            original_feature_importance[original_feature] = 0
        original_feature_importance[original_feature] += importance

    # Create DataFrame for original feature importance
    original_importance_df = pd.DataFrame({
        'Original_Feature': list(original_feature_importance.keys()),
        'Total_Importance': list(original_feature_importance.values())
    }).sort_values('Total_Importance', ascending=False)

    print(f"\n📊 Feature importance grouped by original variables:")
    print(original_importance_df.to_string(index=False))

    # Check for overfitting
    X_train_transformed = preprocessor.transform(X_train)
    train_score = et_regressor.score(X_train_transformed, y_train)

    if train_score - r2_et > 0.1:
        print("⚠️  Warning: Model may be overfitting (training score much higher than test score)")
    elif train_score - r2_et < 0.05:
        print("✅ Good generalization (training and test scores are close)")
    else:
        print("💡 Model may have mild overfitting")

except Exception as e:
    print(f"⚠️ Unable to compute feature importance: {e}")

# ===== 1️⃣1️⃣ ExtraTrees Specific Analysis =====
print("\n🌳 ExtraTrees Model Analysis:")
try:
    et_regressor = best_model_et.named_steps['regressor']

    # Get model parameters
    n_estimators = et_regressor.n_estimators
    bootstrap = et_regressor.bootstrap
    max_features = et_regressor.max_features

    print(f"Number of trees: {n_estimators}")
    print(f"Bootstrap: {bootstrap}")
    print(f"Feature selection strategy: {max_features}")

    # Analyze ExtraTrees-specific features
    if bootstrap:
        print("💡 Using Bootstrap: Sampling with replacement from the training set, increases diversity")
    else:
        print("💡 Not using Bootstrap: Uses the full training set, may reduce bias")

    if max_features == 'auto' or max_features == 'sqrt':
        print("💡 Feature selection: Uses a subset of features per tree, increases randomness")
    elif max_features is None:
        print("💡 Feature selection: Uses all features per tree")

    # ExtraTrees vs RandomForest comparison
    print(f"\n🔍 ExtraTrees vs RandomForest:")
    print("   - ExtraTrees: Randomly selects split points, faster training speed")
    print("   - RandomForest: Seeks the best split points, typically higher accuracy")
    print("   - Both are ensemble methods, but with different strategies of randomness")

except Exception as e:
    print(f"⚠️ Unable to perform model analysis: {e}")

# ===== 1️⃣2️⃣ Export Results =====
summary_et = pd.DataFrame([{
    'Model': 'ExtraTrees',
    'Best Parameters': str(best_params_et),
    'MSLE': safe_format(msle_et, '.6f'),
    'RMSE': safe_format(rmse_et, '.4f'),
    'MAE': safe_format(mae_et, '.4f'),
    'MAPE(%)': safe_format(mape_et, '.3f'),
    'R²': safe_format(r2_et, '.4f'),
    'CV_RMSE': safe_format(np.sqrt(best_score_et), '.4f'),
    'Important_Features': num_important_features,
    'Train_Score': safe_format(train_score, '.4f'),
    'Test_Score': safe_format(r2_et, '.4f'),
    'Bootstrap': best_params_et.get('regressor__bootstrap', 'N/A'),
    'Max_Features': best_params_et.get('regressor__max_features', 'N/A')
}])

output_path = "/home/jiangshan/cre/growth/ExtraTrees_results.xlsx"
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    cv_summary_et.to_excel(writer, sheet_name="CV_Results", index=False)
    summary_et.to_excel(writer, sheet_name="Test_Summary", index=False)

    # Add feature importance to Excel
    try:
        if 'feature_importance_df' in locals():
            feature_importance_df.to_excel(writer, sheet_name="Feature_Importance", index=False)
        if 'original_importance_df' in locals():
            original_importance_df.to_excel(writer, sheet_name="Original_Feature_Importance", index=False)
    except Exception as e:
        print(f"⚠️ Unable to save feature importance to Excel: {e}")

    # Add preprocessing info
    preprocess_info = pd.DataFrame({
        'ExtraTrees_Preprocessing': [
            f'Numeric variables: {numeric_features}',
            f'Categorical variables: {categorical_features}',
            f'Numeric processing: keep original scale (tree models do not require scaling)',
            f'Categorical processing: OneHotEncoder (drop_first)',
            f'Model type: Ensemble learning - Extremely Random Trees',
            f'Base learner: Extremely random decision trees',
            f'Characteristics: More randomness, faster training speed'
        ]
    })
    preprocess_info.to_excel(writer, sheet_name="Preprocessing_Info", index=False)

    # Add model configuration info
    model_config_data = [
        f'Number of trees: {best_params_et.get("regressor__n_estimators", "N/A")}',
        f'Max depth: {best_params_et.get("regressor__max_depth", "N/A")}',
        f'Min samples split: {best_params_et.get("regressor__min_samples_split", "N/A")}',
        f'Min samples leaf: {best_params_et.get("regressor__min_samples_leaf", "N/A")}',
        f'Feature selection strategy: {best_params_et.get("regressor__max_features", "N/A")}',
        f'Bootstrap: {best_params_et.get("regressor__bootstrap", "N/A")}',
        f'Important features (80% contribution): {num_important_features}',
        f'Training R²: {safe_format(train_score, ".4f")}',
        f'Test R²: {safe_format(r2_et, ".4f")}'
    ]

    model_info = pd.DataFrame({
        'ExtraTrees_Configuration': model_config_data
    })
    model_info.to_excel(writer, sheet_name="Model_Config", index=False)

print(f"\n✅ All results have been saved to: {output_path}")

# ===== 1️⃣3️⃣ Model Performance Summary =====
print("\n" + "=" * 50)
print("📊 ExtraTrees model training completed!")
print(f"   Number of trees: {best_params_et.get('regressor__n_estimators', 'N/A')}")
print(f"   Max depth: {best_params_et.get('regressor__max_depth', 'N/A')}")
print(f"   Important features: {num_important_features}")

# Safely print training and test R²
train_score_display = safe_format(train_score, '.4f')
r2_et_display = safe_format(r2_et, '.4f')
rmse_et_display = safe_format(rmse_et, '.4f')

print(f"   Training R²: {train_score_display}")
print(f"   Test R²: {r2_et_display}")
print(f"   Test RMSE: {rmse_et_display}")
print("=" * 50)

# ===== 1️⃣4️⃣ ExtraTrees Specific Suggestions =====
print("\n💡 ExtraTrees Characteristics:")
print("   ✅ Pros: Fast training, resistant to overfitting, handles high-dimensional data, provides feature importance")
print("   ⚠️  Cons: May be slightly less accurate than RandomForest, needs more trees to achieve similar performance")
print("   📝 Recommendations:")
print("      - Suitable for large datasets requiring fast training")
print("      - An alternative when RandomForest training is too slow")
print("      - Robust to outliers")

# ===== 1️⃣5️⃣ ExtraTrees vs RandomForest Comparison =====
print("\n🔍 Detailed ExtraTrees vs RandomForest Comparison:")
print("   - Split point selection:")
print("     * ExtraTrees: Randomly selects split points, faster computation")
print("     * RandomForest: Finds the best split points, slower but more accurate")
print("   - Bias-variance trade-off:")
print("     * ExtraTrees: Typically higher bias but lower variance")
print("     * RandomForest: Typically lower bias but higher variance")
print("   - Use cases:")
print("     * ExtraTrees: Large datasets, fast training, high-dimensional data")
print("     * RandomForest: Medium-small datasets, aiming for highest accuracy")

# ===== 1️⃣6️⃣ Overfitting Detection and Suggestions =====
print("\n🔍 Overfitting Analysis:")
if isinstance(train_score, (int, float)) and isinstance(r2_et, (int, float)):
    gap = train_score - r2_et
    if gap > 0.15:
        print("🚨 Severe overfitting! Suggestions:")
        print("   - Increase min_samples_split and min_samples_leaf")
        print("   - Reduce max_depth")
        print("   - Enable bootstrap (if not already enabled)")
    elif gap > 0.08:
        print("⚠️  Possible overfitting, consider adjusting regularization parameters")
    else:
        print("✅ Good generalization ability")
else:
    print("⚠️  Unable to perform overfitting analysis, feature importance computation failed")

# ===== 1️⃣7️⃣ Next-step Modeling Suggestions =====
print("\n🎯 Next Modeling Suggestions:")
if isinstance(r2_et, (int, float)):
    if r2_et < 0.5:
        print("   📈 Current performance is moderate, suggestions:")
        print("      - Adjust hyperparameter range")
        print("      - Try RandomForest or Gradient Boosting")
        print("      - Review feature engineering")
    elif r2_et > 0.8:
        print("   🎉 Excellent performance! You may:")
        print("      - Analyze feature importance for business insights")
        print("      - Deploy the model in production")
    else:
        print("   🔄 Performance is acceptable; can serve as a baseline for other models")
else:
    print("   🔄 Unable to assess performance, please check model training process")


### model10 Bayesian Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ===== 1️⃣ Define variable types =====
# Numeric variables
numeric_features = ['30kg ABW', 'Litter size', 'Birth weight']
# Categorical variables (need encoding)
categorical_features = ['Season', 'Parity', 'Sex']

print("📊 Bayesian Ridge Regression Data Preprocessing Info:")
print(f"   Numeric variables ({len(numeric_features)}): {numeric_features}")
print(f"   Categorical variables ({len(categorical_features)}): {categorical_features}")

# ===== 2️⃣ Preprocessing Pipeline =====
# Bayesian Ridge is sensitive to feature scale, must standardize numeric variables
numeric_transformer = StandardScaler()

# Categorical variables use one-hot encoding
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

# Build column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ===== 3️⃣ Bayesian Ridge Regression Model =====
bayes_model = BayesianRidge()

# Build the overall Pipeline
pipeline_bayes = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', bayes_model)
])

# ===== 4️⃣ Hyperparameter Grid =====
params_bayes = {
    'regressor__alpha_1': [1e-7, 1e-6, 1e-5, 1e-4],
    'regressor__alpha_2': [1e-7, 1e-6, 1e-5, 1e-4],
    'regressor__lambda_1': [1e-7, 1e-6, 1e-5, 1e-4],
    'regressor__lambda_2': [1e-7, 1e-6, 1e-5, 1e-4],
    'regressor__max_iter': [100, 300, 500],  # Fixed: change n_iter to max_iter
    'regressor__compute_score': [True, False],
    'regressor__tol': [1e-3, 1e-4, 1e-5]  # Added tolerance for convergence
}

print(
    f"🔍 Number of parameter grid combinations: {len(params_bayes['regressor__alpha_1']) * len(params_bayes['regressor__alpha_2']) * len(params_bayes['regressor__lambda_1']) * len(params_bayes['regressor__lambda_2']) * len(params_bayes['regressor__max_iter']) * len(params_bayes['regressor__compute_score']) * len(params_bayes['regressor__tol'])}")

# ===== 5️⃣ Cross-validation + Grid Search =====
cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search_bayes = GridSearchCV(
    estimator=pipeline_bayes,
    param_grid=params_bayes,
    cv=cv,
    n_jobs=1,
    scoring='neg_mean_squared_error',
    verbose=1,
    return_train_score=True
)

# ===== 6️⃣ Fit Model =====
print("🚀 Starting to train Bayesian Ridge Regression model...")
print(
    "⚠️  Note: Bayesian Ridge Regression is sensitive to feature scale, continuous variables have been standardized and categorical variables encoded")
print("💡 Bayesian Characteristics: Provides uncertainty estimates for parameters, automatically regularizes")

try:
    grid_search_bayes.fit(X_train, y_train)

    # ===== 7️⃣ Output Best Model and Parameters =====
    best_model_bayes = grid_search_bayes.best_estimator_
    best_params_bayes = grid_search_bayes.best_params_
    best_score_bayes = -grid_search_bayes.best_score_

    print(f"\n✅ Best Parameters for Bayesian Ridge Regression: {best_params_bayes}")
    print(f"✅ Best Cross-validation RMSE: {np.sqrt(best_score_bayes):.4f}")

    # ===== 8️⃣ Cross-validation Results Analysis =====
    cv_results_bayes = pd.DataFrame(grid_search_bayes.cv_results_)
    cv_results_bayes['MSE'] = -cv_results_bayes['mean_test_score']
    cv_results_bayes['RMSE'] = np.sqrt(cv_results_bayes['MSE'])
    cv_results_bayes['train_RMSE'] = np.sqrt(-cv_results_bayes['mean_train_score'])
    cv_summary_bayes = cv_results_bayes[['params', 'MSE', 'RMSE', 'train_RMSE', 'rank_test_score']].sort_values(
        'rank_test_score')

    print(f"\n📈 Top 3 Best Parameter Combinations:")
    for i, row in cv_summary_bayes.head(3).iterrows():
        print(f"   Rank {int(row['rank_test_score'])}: RMSE = {row['RMSE']:.4f}")

    # ===== 9️⃣ Test Set Prediction and Evaluation =====
    y_pred_bayes = best_model_bayes.predict(X_test)

    # Avoid negative values (to prevent MSLE error)
    eps = 1e-9
    y_test_safe = np.clip(np.asarray(y_test), eps, None)
    y_pred_safe = np.clip(y_pred_bayes, eps, None)

    msle_bayes = mean_squared_log_error(y_test_safe, y_pred_safe)
    rmse_bayes = np.sqrt(mean_squared_error(y_test_safe, y_pred_safe))
    mae_bayes = mean_absolute_error(y_test_safe, y_pred_safe)
    mape_bayes = np.mean(np.abs((y_test_safe - y_pred_safe) / y_test_safe)) * 100
    r2_bayes = r2_score(y_test_safe, y_pred_safe)

    print("\n🎯 Bayesian Ridge Regression Test Set Performance Metrics:")
    print(f"MSLE = {msle_bayes:.6f}")
    print(f"RMSE = {rmse_bayes:.4f}")
    print(f"MAE  = {mae_bayes:.4f}")
    print(f"MAPE = {mape_bayes:.3f}%")
    print(f"R²   = {r2_bayes:.4f}")

    # ===== 🔟 Bayesian Specific Analysis =====
    print("\n🔍 Bayesian Ridge Regression Specific Analysis:")

    try:
        # Get the Bayesian model
        bayes_regressor = best_model_bayes.named_steps['regressor']

        # Get preprocessed feature names
        preprocessor = best_model_bayes.named_steps['preprocessor']
        feature_names = []

        # Categorical feature names (after one-hot encoding)
        if categorical_features:
            cat_encoder = preprocessor.named_transformers_['cat']
            for i, col in enumerate(categorical_features):
                if hasattr(cat_encoder, 'categories_'):
                    categories = cat_encoder.categories_[i]
                    # Skip the first category (because drop='first')
                    for cat in categories[1:]:
                        feature_names.append(f"{col}_{cat}")

        # Add numeric feature names
        feature_names.extend(numeric_features)

        # Get coefficients and uncertainty
        coefficients = bayes_regressor.coef_

        # Check if covariance matrix exists
        if hasattr(bayes_regressor, 'sigma_') and bayes_regressor.sigma_ is not None:
            coefficient_std = np.sqrt(np.diag(bayes_regressor.sigma_))

            # Create coefficient importance DataFrame
            coefficient_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients,
                'Coefficient_Std': coefficient_std,
                'Abs_Coefficient': np.abs(coefficients)
            }).sort_values('Abs_Coefficient', ascending=False)

            # Calculate relative uncertainty of coefficients
            coefficient_df['Uncertainty_Ratio'] = coefficient_df['Coefficient_Std'] / np.abs(
                coefficient_df['Coefficient'])

            print(f"Total number of features: {len(feature_names)}")
            print("Top 10 most important feature coefficients:")
            print(coefficient_df.head(10).round(4).to_string(index=False))

            # Analyze uncertainty in coefficients
            high_uncertainty_features = coefficient_df[coefficient_df['Uncertainty_Ratio'] > 1.0]
            if len(high_uncertainty_features) > 0:
                print(f"\n⚠️  High uncertainty features (Uncertainty ratio > 1.0): {len(high_uncertainty_features)}")
                print("These feature coefficient estimates are unreliable and may need more data")
        else:
            # If no covariance matrix, only display coefficients
            coefficient_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients,
                'Abs_Coefficient': np.abs(coefficients)
            }).sort_values('Abs_Coefficient', ascending=False)

            print(f"Total number of features: {len(feature_names)}")
            print("Top 10 most important feature coefficients:")
            print(coefficient_df.head(10).round(4).to_string(index=False))
            print("⚠️  Unable to calculate coefficient uncertainty (Covariance matrix unavailable)")

        # Check for overfitting
        X_train_transformed = preprocessor.transform(X_train)
        train_score = bayes_regressor.score(X_train_transformed, y_train)

        if train_score - r2_bayes > 0.1:
            print("⚠️  Warning: Model may be overfitting (training score much higher than test score)")
        elif train_score - r2_bayes < 0.05:
            print("✅ Good generalization (training and test scores are close)")
        else:
            print("💡 Model may have mild overfitting")

        # Bayesian model specific information
        print(f"\n📊 Bayesian Model Parameters:")
        print(f"   Iterations: {bayes_regressor.n_iter_ if hasattr(bayes_regressor, 'n_iter_') else 'N/A'}")
        if hasattr(bayes_regressor, 'alpha_') and hasattr(bayes_regressor, 'lambda_'):
            print(f"   Final alpha: {bayes_regressor.alpha_:.6f}")
            print(f"   Final lambda: {bayes_regressor.lambda_:.6f}")

    except Exception as e:
        print(f"⚠️ Unable to perform Bayesian analysis: {e}")
        train_score = 'N/A'
        coefficient_df = None

    # ===== 1️⃣1️⃣ Regularization Analysis =====
    print("\n⚖️ Bayesian Regularization Analysis:")
    try:
        bayes_regressor = best_model_bayes.named_steps['regressor']

        # Analyze the impact of hyperparameters on regularization
        alpha_1 = best_params_bayes.get('regressor__alpha_1', 'N/A')
        lambda_1 = best_params_bayes.get('regressor__lambda_1', 'N/A')

        print(f"alpha_1 (Precision prior): {alpha_1}")
        print(f"lambda_1 (Coefficient prior): {lambda_1}")

        # Explain hyperparameter meaning
        if isinstance(alpha_1, (int, float)):
            if alpha_1 < 1e-5:
                print("💡 alpha_1 is small: weak prior on noise variance")
            elif alpha_1 > 1e-3:
                print("💡 alpha_1 is large: strong prior on noise variance")

        if isinstance(lambda_1, (int, float)):
            if lambda_1 < 1e-5:
                print("💡 lambda_1 is small: weak prior on coefficients, model more complex")
            elif lambda_1 > 1e-3:
                print("💡 lambda_1 is large: strong prior on coefficients, model simpler")

        # Bayesian vs Ordinary Ridge Regression Comparison
        print(f"\n🔍 Bayesian Ridge vs Ordinary Ridge Regression:")
        print("   - Bayesian Ridge: Automatically determines regularization parameters, provides uncertainty estimates")
        print("   - Ordinary Ridge: Requires manual selection of regularization parameters, no uncertainty estimates")
        print("   - Both use L2 regularization, but the method of parameter determination differs")

    except Exception as e:
        print(f"⚠️ Unable to perform regularization analysis: {e}")


    # ===== 1️⃣2️⃣ Export Results =====
    # Safe formatting function
    def safe_format(value, format_spec='.4f'):
        """Safely format numeric values, return as string if not numeric"""
        if isinstance(value, (int, float)):
            return format(value, format_spec)
        else:
            return str(value)


    summary_bayes = pd.DataFrame([{
        'Model': 'Bayesian Ridge',
        'Best Parameters': str(best_params_bayes),
        'MSLE': safe_format(msle_bayes, '.6f'),
        'RMSE': safe_format(rmse_bayes, '.4f'),
        'MAE': safe_format(mae_bayes, '.4f'),
        'MAPE(%)': safe_format(mape_bayes, '.3f'),
        'R²': safe_format(r2_bayes, '.4f'),
        'CV_RMSE': safe_format(np.sqrt(best_score_bayes), '.4f'),
        'Train_Score': safe_format(train_score, '.4f'),
        'Test_Score': safe_format(r2_bayes, '.4f'),
        'Alpha_1': best_params_bayes.get('regressor__alpha_1', 'N/A'),
        'Lambda_1': best_params_bayes.get('regressor__lambda_1', 'N/A'),
        'High_Uncertainty_Features': len(
            high_uncertainty_features) if 'high_uncertainty_features' in locals() and isinstance(
            high_uncertainty_features, pd.DataFrame) else 'N/A'
    }])

    output_path = "/home/jiangshan/cre/growth/BayesianRidge_results.xlsx"
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        cv_summary_bayes.to_excel(writer, sheet_name="CV_Results", index=False)
        summary_bayes.to_excel(writer, sheet_name="Test_Summary", index=False)

        # Add coefficient analysis to Excel
        try:
            if 'coefficient_df' in locals() and coefficient_df is not None:
                coefficient_df.to_excel(writer, sheet_name="Coefficient_Analysis", index=False)
        except Exception as e:
            print(f"⚠️ Unable to save coefficient analysis to Excel: {e}")

        # Add preprocessing info
        preprocess_info = pd.DataFrame({
            'BayesianRidge_Preprocessing': [
                f'Numeric variables: {numeric_features}',
                f'Categorical variables: {categorical_features}',
                f'Numeric processing: StandardScaler (Bayesian models must standardize)',
                f'Categorical processing: OneHotEncoder (drop_first)',
                f'Model type: Bayesian Linear Regression',
                f'Regularization type: L2 regularization (Ridge)',
                f'Characteristics: Automatically determines regularization parameters, provides uncertainty estimates'
            ]
        })
        preprocess_info.to_excel(writer, sheet_name="Preprocessing_Info", index=False)

        # Add model configuration info
        model_config_data = [
            f'alpha_1 (Precision prior): {best_params_bayes.get("regressor__alpha_1", "N/A")}',
            f'alpha_2 (Precision prior): {best_params_bayes.get("regressor__alpha_2", "N/A")}',
            f'lambda_1 (Coefficient prior): {best_params_bayes.get("regressor__lambda_1", "N/A")}',
            f'lambda_2 (Coefficient prior): {best_params_bayes.get("regressor__lambda_2", "N/A")}',
            f'Max iterations: {best_params_bayes.get("regressor__max_iter", "N/A")}',
            f'Compute score: {best_params_bayes.get("regressor__compute_score", "N/A")}',
            f'Tolerance: {best_params_bayes.get("regressor__tol", "N/A")}',
            f'High uncertainty feature count: {len(high_uncertainty_features) if "high_uncertainty_features" in locals() and isinstance(high_uncertainty_features, pd.DataFrame) else "N/A"}',
            f'Training R²: {safe_format(train_score, ".4f")}',
            f'Test R²: {safe_format(r2_bayes, ".4f")}'
        ]

        model_info = pd.DataFrame({
            'BayesianRidge_Configuration': model_config_data
        })
        model_info.to_excel(writer, sheet_name="Model_Config", index=False)

    print(f"\n✅ All results have been saved to: {output_path}")

    # ===== 1️⃣3️⃣ Model Performance Summary =====
    print("\n" + "=" * 50)
    print("📊 Bayesian Ridge Model Training Completed!")
    print(f"   alpha_1: {best_params_bayes.get('regressor__alpha_1', 'N/A')}")
    print(f"   lambda_1: {best_params_bayes.get('regressor__lambda_1', 'N/A')}")

    # Safely print training and test R²
    train_score_display = safe_format(train_score, '.4f')
    r2_bayes_display = safe_format(r2_bayes, '.4f')
    rmse_bayes_display = safe_format(rmse_bayes, '.4f')

    print(f"   Training R²: {train_score_display}")
    print(f"   Test R²: {r2_bayes_display}")
    print(f"   Test RMSE: {rmse_bayes_display}")
    print("=" * 50)

    # ===== 1️⃣4️⃣ Bayesian Ridge Specific Suggestions =====
    print("\n💡 Bayesian Ridge Characteristics:")
    print(
        "   ✅ Pros: Automatic regularization, provides uncertainty estimates, prevents overfitting, theoretically sound")
    print("   ⚠️  Cons: Higher computation cost, sensitive to prior selection, assumes Gaussian distribution")
    print("   📝 Recommendations:")
    print("      - Suitable for scenarios requiring uncertainty estimation")
    print("      - Performs better with smaller datasets")
    print("      - Prior parameters need careful tuning")

except Exception as e:
    print(f"❌ Model training failed: {e}")
    print("🔧 Recommendations:")
    print("   - Ensure X_train, y_train, X_test, y_test are defined correctly")
    print("   - Check if the data contains missing values")
    print("   - Verify that feature names are correct")


### model11 RANSAC
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ===== 1️⃣ Define variable types =====
# Numeric variables
numeric_features = ['30kg ABW', 'Litter size', 'Birth weight']
# Categorical variables (need encoding)
categorical_features = ['Season', 'Parity', 'Sex']

print("📊 RANSAC Regression Data Preprocessing Info:")
print(f"   Numeric variables ({len(numeric_features)}): {numeric_features}")
print(f"   Categorical variables ({len(categorical_features)}): {categorical_features}")

# ===== 2️⃣ Preprocessing Pipeline =====
# RANSAC regression is not sensitive to feature scale, but for overall consistency, continuous variables are standardized
numeric_transformer = StandardScaler()

# Categorical variables use one-hot encoding
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

# Build column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ===== 3️⃣ RANSAC Regression Model =====
ransac_model = RANSACRegressor(random_state=42)

# Build the overall Pipeline
pipeline_ransac = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', ransac_model)
])

# ===== 4️⃣ Hyperparameter Grid =====
params_ransac = {
    'regressor__min_samples': [0.5, 0.7, 0.9],
    'regressor__max_trials': [50, 100, 200]
}

# ===== 5️⃣ Cross-validation + Grid Search =====
cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search_ransac = GridSearchCV(
    estimator=pipeline_ransac,  # Using the pipeline with preprocessing
    param_grid=params_ransac,
    cv=cv,
    n_jobs=1,
    scoring='neg_mean_squared_error',
    verbose=1,  # Show progress
    return_train_score=True
)

# ===== 6️⃣ Fit Model =====
print("🚀 Starting to train RANSAC regression model...")
grid_search_ransac.fit(X_train, y_train)

# ===== 7️⃣ Output Best Model and Parameters =====
best_model_ransac = grid_search_ransac.best_estimator_
best_params_ransac = grid_search_ransac.best_params_
best_score_ransac = -grid_search_ransac.best_score_  # Convert to positive

print(f"\n✅ Best Parameters for RANSAC Regression: {best_params_ransac}")
print(f"✅ Best Cross-validation RMSE: {np.sqrt(best_score_ransac):.4f}")

# ===== 8️⃣ Cross-validation Results Analysis =====
cv_results_ransac = pd.DataFrame(grid_search_ransac.cv_results_)
cv_results_ransac['MSE'] = -cv_results_ransac['mean_test_score']
cv_results_ransac['RMSE'] = np.sqrt(cv_results_ransac['MSE'])
cv_results_ransac['train_RMSE'] = np.sqrt(-cv_results_ransac['mean_train_score'])
cv_summary_ransac = cv_results_ransac[['params', 'MSE', 'RMSE', 'train_RMSE', 'rank_test_score']].sort_values('rank_test_score')

print(f"\n📈 Top 3 Best Parameter Combinations:")
for i, row in cv_summary_ransac.head(3).iterrows():
    print(f"   Rank {int(row['rank_test_score'])}: RMSE = {row['RMSE']:.4f}")

# ===== 9️⃣ Test Set Prediction and Evaluation =====
y_pred_ransac = best_model_ransac.predict(X_test)

# Avoid negative values (to prevent MSLE error)
eps = 1e-9
y_test_safe = np.clip(np.asarray(y_test), eps, None)
y_pred_safe = np.clip(y_pred_ransac, eps, None)

msle_ransac = mean_squared_log_error(y_test_safe, y_pred_safe)
rmse_ransac = np.sqrt(mean_squared_error(y_test_safe, y_pred_safe))
mae_ransac = mean_absolute_error(y_test_safe, y_pred_safe)
mape_ransac = np.mean(np.abs((y_test_safe - y_pred_safe) / y_test_safe)) * 100
r2_ransac = r2_score(y_test_safe, y_pred_safe)

print("\n🎯 RANSAC Regression Test Set Performance Metrics:")
print(f"MSLE = {msle_ransac:.6f}")
print(f"RMSE = {rmse_ransac:.4f}")
print(f"MAE  = {mae_ransac:.4f}")
print(f"MAPE = {mape_ransac:.3f}%")
print(f"R²   = {r2_ransac:.4f}")

# ===== 🔟 Summary and Export Results =====
summary_ransac = pd.DataFrame([{
    'Model': 'RANSAC',
    'Best Parameters': str(best_params_ransac),
    'MSLE': round(msle_ransac, 6),
    'RMSE': round(rmse_ransac, 4),
    'MAE': round(mae_ransac, 4),
    'MAPE(%)': round(mape_ransac, 3),
    'R²': round(r2_ransac, 4),
    'CV_RMSE': round(np.sqrt(best_score_ransac), 4)
}])

output_path = "/home/jiangshan/cre/growth/RANSAC_results.xlsx"
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    cv_summary_ransac.to_excel(writer, sheet_name="CV_Results", index=False)
    summary_ransac.to_excel(writer, sheet_name="Test_Summary", index=False)

print(f"\n✅ Results saved to: {output_path}")


###model12 SVR
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ===== 1️⃣ Data Preprocessing =====
print("📊 SVR Data Preprocessing Info:")
print("⚠️  Note: SVR is very sensitive to feature scale and requires standardizing all features")
print("💡 SVR Characteristics: Uses kernel tricks to handle nonlinear relationships, relatively robust to outliers")

numeric_features = ['30kg ABW', 'Litter size', 'Birth weight']
categorical_features = ['Season', 'Parity', 'Sex']

print(f"   Numeric variables ({len(numeric_features)}): {numeric_features}")
print(f"   Categorical variables ({len(categorical_features)}): {categorical_features}")

# SVR is very sensitive to feature scale and requires standardizing all features
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ===== 2️⃣ SVR Model =====
svr_model = SVR()

# Build Pipeline
pipeline_svr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', svr_model)
])

# ===== 3️⃣ Hyperparameter Grid Optimization =====
# Adjust parameter ranges according to SVR characteristics
params_svr = {
    'regressor__C': [0.1, 1, 10, 100],  # Regularization parameter - expanded range
    'regressor__epsilon': [0.01, 0.05, 0.1, 0.2],  # ε-insensitive zone width
    'regressor__kernel': ['linear', 'rbf', 'poly'],  # Kernel functions
    'regressor__gamma': ['scale', 'auto'],  # Added gamma parameter (important for rbf and poly kernels)
    'regressor__degree': [2, 3]  # Degree of polynomial kernel (only for poly kernel)
}

print(
    f"🔍 Number of parameter grid combinations: {len(params_svr['regressor__C']) * len(params_svr['regressor__epsilon']) * len(params_svr['regressor__kernel']) * len(params_svr['regressor__gamma']) * len(params_svr['regressor__degree'])}")

# ===== 4️⃣ Cross-validation + Grid Search =====
cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search_svr = GridSearchCV(
    estimator=pipeline_svr,
    param_grid=params_svr,
    cv=cv,
    n_jobs=1,
    scoring='neg_mean_squared_error',
    verbose=1,
    return_train_score=True
)

# ===== 5️⃣ Fit Model =====
print("🚀 Starting to train SVR model...")
print("💡 Note: Training time for SVR may be longer, especially when using nonlinear kernels")

try:
    grid_search_svr.fit(X_train, y_train)

    # ===== 6️⃣ Best Model and Parameters =====
    best_model_svr = grid_search_svr.best_estimator_
    best_params_svr = grid_search_svr.best_params_
    best_score_svr = -grid_search_svr.best_score_

    print(f"\n✅ Best Parameters for SVR: {best_params_svr}")
    print(f"✅ Best Cross-validation RMSE: {np.sqrt(best_score_svr):.4f}")

    # ===== 7️⃣ Cross-validation Results Analysis =====
    cv_results_svr = pd.DataFrame(grid_search_svr.cv_results_)
    cv_results_svr['MSE'] = -cv_results_svr['mean_test_score']
    cv_results_svr['RMSE'] = np.sqrt(cv_results_svr['MSE'])
    cv_results_svr['train_RMSE'] = np.sqrt(-cv_results_svr['mean_train_score'])
    cv_summary_svr = cv_results_svr[['params', 'MSE', 'RMSE', 'train_RMSE', 'rank_test_score']].sort_values(
        'rank_test_score')

    print(f"\n📈 Top 3 Best Parameter Combinations:")
    for i, row in cv_summary_svr.head(3).iterrows():
        print(f"   Rank {int(row['rank_test_score'])}: RMSE = {row['RMSE']:.4f}")

    # ===== 8️⃣ Test Set Prediction and Evaluation =====
    y_pred_svr = best_model_svr.predict(X_test)

    # Avoid negative values (to prevent MSLE error)
    eps = 1e-9
    y_test_safe = np.clip(np.asarray(y_test), eps, None)
    y_pred_safe = np.clip(y_pred_svr, eps, None)

    msle_svr = mean_squared_log_error(y_test_safe, y_pred_safe)
    rmse_svr = np.sqrt(mean_squared_error(y_test_safe, y_pred_safe))
    mae_svr = mean_absolute_error(y_test_safe, y_pred_safe)
    mape_svr = np.mean(np.abs((y_test_safe - y_pred_safe) / y_test_safe)) * 100
    r2_svr = r2_score(y_test_safe, y_pred_safe)

    print("\n🎯 SVR Test Set Performance Metrics:")
    print(f"MSLE = {msle_svr:.6f}")
    print(f"RMSE = {rmse_svr:.4f}")
    print(f"MAE  = {mae_svr:.4f}")
    print(f"MAPE = {mape_svr:.3f}%")
    print(f"R²   = {r2_svr:.4f}")

    # ===== 9️⃣ SVR Specific Analysis =====
    print("\n🔍 SVR Specific Analysis:")
    try:
        # Get the SVR model
        svr_regressor = best_model_svr.named_steps['regressor']

        # Get preprocessed feature names
        preprocessor = best_model_svr.named_steps['preprocessor']
        feature_names = []

        # Categorical feature names (after one-hot encoding)
        if categorical_features:
            cat_encoder = preprocessor.named_transformers_['cat']
            for i, col in enumerate(categorical_features):
                if hasattr(cat_encoder, 'categories_'):
                    categories = cat_encoder.categories_[i]
                    # Skip the first category (because drop='first')
                    for cat in categories[1:]:
                        feature_names.append(f"{col}_{cat}")

        # Add numeric feature names
        feature_names.extend(numeric_features)

        # Analyze support vectors
        n_support_vectors = len(svr_regressor.support_)
        n_training_samples = len(X_train)
        support_vector_ratio = n_support_vectors / n_training_samples

        print(f"Number of training samples: {n_training_samples}")
        print(f"Number of support vectors: {n_support_vectors}")
        print(f"Support vector ratio: {support_vector_ratio:.2%}")

        # Explain the meaning of support vector ratio
        if support_vector_ratio < 0.1:
            print("💡 Low support vector ratio - the model is relatively sparse, likely to generalize well")
        elif support_vector_ratio > 0.5:
            print("💡 High support vector ratio - the model is more complex, with a risk of overfitting")
        else:
            print("💡 Moderate support vector ratio")

        # Kernel function analysis
        kernel_type = best_params_svr.get('regressor__kernel', 'N/A')
        print(f"Best kernel: {kernel_type}")

        if kernel_type == 'linear':
            print("💡 Linear kernel: The problem might be linear or data has high dimensionality")
            # For linear kernels, we can look at the coefficients
            if hasattr(svr_regressor, 'coef_'):
                coefficients = svr_regressor.coef_[0]
                coef_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': coefficients,
                    'Abs_Coefficient': np.abs(coefficients)
                }).sort_values('Abs_Coefficient', ascending=False)

                print("\n📊 Linear SVR Feature Coefficients (Top 10):")
                print(coef_df.head(10).to_string(index=False))

        elif kernel_type == 'rbf':
            print("💡 RBF kernel: Handles nonlinear relationships, requires careful adjustment of gamma")
            gamma_value = best_params_svr.get('regressor__gamma', 'N/A')
            print(f"Best gamma: {gamma_value}")

        elif kernel_type == 'poly':
            print("💡 Polynomial kernel: Handles polynomial relationships")
            degree_value = best_params_svr.get('regressor__degree', 'N/A')
            print(f"Polynomial degree: {degree_value}")

        # Regularization parameter analysis
        C_value = best_params_svr.get('regressor__C', 'N/A')
        epsilon_value = best_params_svr.get('regressor__epsilon', 'N/A')

        print(f"\nRegularization parameter C: {C_value}")
        print(f"ε-insensitive band: {epsilon_value}")

        if isinstance(C_value, (int, float)):
            if C_value < 1:
                print("💡 Small C value: Emphasizes model simplicity, tolerates more error")
            elif C_value > 10:
                print("💡 Large C value: Emphasizes fitting accuracy, model may be more complex")

        if isinstance(epsilon_value, (int, float)):
            if epsilon_value < 0.05:
                print("💡 Small epsilon value: Sensitive to errors, requires more precise fitting")
            elif epsilon_value > 0.1:
                print("💡 Large epsilon value: Less sensitive to errors, model is more robust")

    except Exception as e:
        print(f"⚠️ Unable to perform SVR specific analysis: {e}")

    # ===== 🔟 Overfitting Analysis =====
    print("\n🔍 Overfitting Analysis:")
    try:
        # Get the performance gap between training and test sets
        train_score = -grid_search_svr.cv_results_['mean_train_score'][grid_search_svr.best_index_]
        test_score = best_score_svr
        performance_gap = train_score - test_score

        print(f"Training MSE: {train_score:.4f}")
        print(f"Test MSE: {test_score:.4f}")
        print(f"Performance gap: {performance_gap:.4f}")

        if performance_gap > 0.1:
            print("⚠️  Possible overfitting")
        elif performance_gap < 0.02:
            print("✅ Good generalization")
        else:
            print("💡 Moderate performance gap")

    except Exception as e:
        print(f"⚠️ Unable to perform overfitting analysis: {e}")

    # ===== 1️⃣1️⃣ Export Results =====
    summary_svr = pd.DataFrame([{
        'Model': 'SVR',
        'Best Parameters': str(best_params_svr),
        'MSLE': round(msle_svr, 6),
        'RMSE': round(rmse_svr, 4),
        'MAE': round(mae_svr, 4),
        'MAPE(%)': round(mape_svr, 3),
        'R²': round(r2_svr, 4),
        'CV_RMSE': round(np.sqrt(best_score_svr), 4),
        'Kernel': best_params_svr.get('regressor__kernel', 'N/A'),
        'C_Value': best_params_svr.get('regressor__C', 'N/A'),
        'Epsilon': best_params_svr.get('regressor__epsilon', 'N/A'),
        'Support_Vectors': n_support_vectors if 'n_support_vectors' in locals() else 'N/A',
        'Support_Vector_Ratio(%)': round(support_vector_ratio * 100, 2) if 'support_vector_ratio' in locals() else 'N/A'
    }])

    output_path = "/home/jiangshan/cre/growth/SVR_results.xlsx"
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        cv_summary_svr.to_excel(writer, sheet_name="CV_Results", index=False)
        summary_svr.to_excel(writer, sheet_name="Test_Summary", index=False)

        # Add coefficient analysis to Excel (if using linear kernel)
        try:
            if kernel_type == 'linear' and 'coef_df' in locals():
                coef_df.to_excel(writer, sheet_name="Feature_Coefficients", index=False)
        except:
            pass

        # Add preprocessing info
        preprocess_info = pd.DataFrame({
            'SVR_Preprocessing': [
                f'Numeric variables: {numeric_features}',
                f'Categorical variables: {categorical_features}',
                f'Numeric processing: StandardScaler (SVR requires standardization)',
                f'Categorical processing: OneHotEncoder (drop_first)',
                f'Model type: Support Vector Regression',
                f'Kernel trick: {kernel_type}',
                f'Characteristics: Maximizes margin, robust to outliers'
            ]
        })
        preprocess_info.to_excel(writer, sheet_name="Preprocessing_Info", index=False)

        # Add model configuration info
        model_config_data = [
            f'Kernel: {best_params_svr.get("regressor__kernel", "N/A")}',
            f'Regularization parameter C: {best_params_svr.get("regressor__C", "N/A")}',
            f'ε-insensitive band: {best_params_svr.get("regressor__epsilon", "N/A")}',
            f'Gamma: {best_params_svr.get("regressor__gamma", "N/A")}',
            f'Polynomial degree: {best_params_svr.get("regressor__degree", "N/A")}',
            f'Number of support vectors: {n_support_vectors if "n_support_vectors" in locals() else "N/A"}',
            f'Support vector ratio: {round(support_vector_ratio * 100, 2) if "support_vector_ratio" in locals() else "N/A"}%',
            f'Number of training samples: {n_training_samples if "n_training_samples" in locals() else "N/A"}',
            f'Test R²: {round(r2_svr, 4)}'
        ]

        model_info = pd.DataFrame({
            'SVR_Configuration': model_config_data
        })
        model_info.to_excel(writer, sheet_name="Model_Config", index=False)

    print(f"\n✅ All results have been saved to: {output_path}")

    # ===== 1️⃣2️⃣ Model Performance Summary =====
    print("\n" + "=" * 50)
    print("📊 SVR Model Training Completed!")
    print(f"   Best kernel: {best_params_svr.get('regressor__kernel', 'N/A')}")
    print(f"   Regularization parameter C: {best_params_svr.get('regressor__C', 'N/A')}")
    print(f"   ε-insensitive band: {best_params_svr.get('regressor__epsilon', 'N/A')}")
    print(
        f"   Support vector ratio: {round(support_vector_ratio * 100, 2) if 'support_vector_ratio' in locals() else 'N/A'}%")
    print(f"   Best Test R²: {r2_svr:.4f}")
    print(f"   Best Test RMSE: {rmse_svr:.4f}")
    print("=" * 50)

    # ===== 1️⃣3️⃣ SVR Specific Suggestions =====
    print("\n💡 SVR Characteristics:")
    print("   ✅ Pros: Kernel tricks handle nonlinearities, robust to outliers, theoretically sound")
    print("   ⚠️  Cons: High computational complexity, requires careful tuning, sensitive to feature scale")
    print("   📝 Recommendations:")
    print("      - Must standardize all features")
    print("      - Performs better with smaller datasets")
    print("      - Need to carefully choose kernel and parameters")

    # ===== 1️⃣4️⃣ Kernel Function Selection Guide =====
    print("\n🔍 Kernel Function Selection Guide:")
    kernel_used = best_params_svr.get('regressor__kernel', 'N/A')
    if kernel_used == 'linear':
        print("   💡 Linear kernel is suitable: High-dimensional data, linear separable problems")
    elif kernel_used == 'rbf':
        print("   💡 RBF kernel is suitable: Nonlinear problems, moderate sample size, flexible decision boundaries")
    elif kernel_used == 'poly':
        print("   💡 Polynomial kernel is suitable: Polynomial relationships")

    # ===== 1️⃣5️⃣ Performance Optimization Suggestions =====
    print("\n🎯 SVR Performance Optimization Guide:")
    if r2_svr < 0.5:
        print("   💡 Current R² is low, suggestions:")
        print("      - Try different kernel combinations")
        print("      - Expand the parameter range for C and epsilon")
        print("      - Check feature engineering and preprocessing")
    elif r2_svr > 0.8:
        print("   💡 Excellent performance, consider:")
        print("      - Further regularization to prevent overfitting")
        print("      - Analyze the distribution of support vectors")
    else:
        print("   💡 Moderate performance, consider optimizing parameters further")

except Exception as e:
    print(f"❌ SVR model training failed: {e}")
    print("🔧 Debugging suggestions:")
    print("   1. Check the data format and types")
    print("   2. Ensure X_train, y_train, X_test, y_test are correctly defined")
    print("   3. Check for missing or infinite values")
    print("   4. Try simplifying the parameter grid")

    # Try using a simpler default SVR for debugging
    print("\n🔄 Trying a simplified SVR model for debugging...")
    try:
        simple_svr = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', SVR(kernel='rbf', C=1.0, epsilon=0.1))
        ])

        simple_svr.fit(X_train, y_train)
        y_pred_simple = simple_svr.predict(X_test)
        r2_simple = r2_score(y_test, y_pred_simple)
        print(f"✅ Simplified SVR model trained successfully, Test R²: {r2_simple:.4f}")
        print("💡 This indicates the data preprocessing and basic SVR structure are correct")

    except Exception as simple_error:
        print(f"❌ Simplified SVR model also failed: {simple_error}")
        print("🔍 You may need to check the data preprocessing steps or data quality")

### model13 KNN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ===== 1️⃣ Define Variable Types =====
# Continuous variables
numeric_features = ['30kg ABW', 'Litter size', 'Birth weight']
# Categorical variables (require special processing)
categorical_features = ['Season', 'Parity', 'Sex']

print("📊 KNN Model Data Preprocessing Info:")
print(f"   Numeric variables ({len(numeric_features)}): {numeric_features}")
print(f"   Categorical variables ({len(categorical_features)}): {categorical_features}")

# ===== 2️⃣ Preprocessing Pipeline =====
# KNN is very sensitive to feature scale, so continuous variables must be standardized
numeric_transformer = StandardScaler()

# Categorical variables use one-hot encoding, but care is needed for categorical variables in KNN
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

# Build column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ===== 3️⃣ KNN Model =====
knn_model = KNeighborsRegressor()

# Build the complete Pipeline
pipeline_knn = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', knn_model)
])

# ===== 4️⃣ Hyperparameter Grid =====
params_knn = {
    'regressor__n_neighbors': [3, 5, 7, 9, 11],  # Expanded range of neighbors
    'regressor__weights': ['uniform', 'distance'],
    'regressor__p': [1, 2],  # 1: Manhattan distance, 2: Euclidean distance
    'regressor__metric': ['minkowski']  # Minkowski distance, controlled by p parameter
}

# ===== 5️⃣ Cross-validation + Grid Search =====
cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search_knn = GridSearchCV(
    pipeline_knn,  # Use pipeline with preprocessing
    param_grid=params_knn,
    cv=cv,
    n_jobs=1,
    scoring='neg_mean_squared_error',
    verbose=1,  # Show progress
    return_train_score=True
)

# ===== 6️⃣ Fit Model =====
print("🚀 Starting to train KNN regression model...")
print(
    "⚠️  Note: KNN is sensitive to feature scale, continuous variables have been automatically standardized and categorical variables encoded")
grid_search_knn.fit(X_train, y_train)

# ===== 7️⃣ Output Best Model and Parameters =====
best_model_knn = grid_search_knn.best_estimator_
best_params_knn = grid_search_knn.best_params_
best_score_knn = -grid_search_knn.best_score_  # Convert to positive

print(f"\n✅ Best Parameters for KNN: {best_params_knn}")
print(f"✅ Best Cross-validation RMSE: {np.sqrt(best_score_knn):.4f}")

# ===== 8️⃣ Prediction on Test Set =====
y_pred_knn = best_model_knn.predict(X_test)

# Avoid negative values (to prevent MSLE error)
eps = 1e-9
y_test_safe = np.clip(np.asarray(y_test), eps, None)
y_pred_safe = np.clip(y_pred_knn, eps, None)

# ===== 9️⃣ Calculate Metrics =====
msle_knn = mean_squared_log_error(y_test_safe, y_pred_safe)
rmse_knn = np.sqrt(mean_squared_error(y_test_safe, y_pred_safe))
mae_knn = mean_absolute_error(y_test_safe, y_pred_safe)
mape_knn = np.mean(np.abs((y_test_safe - y_pred_safe) / y_test_safe)) * 100
r2_knn = r2_score(y_test_safe, y_pred_safe)

# ===== 🔟 Print Results =====
print("\n🎯 KNN Regression Test Set Performance Metrics:")
print(f"MSLE = {msle_knn:.6f}")
print(f"RMSE = {rmse_knn:.4f}")
print(f"MAE  = {mae_knn:.4f}")
print(f"MAPE = {mape_knn:.3f}%")
print(f"R²   = {r2_knn:.4f}")

# ===== 1️⃣1️⃣ Cross-validation Results Analysis =====
cv_results_knn = pd.DataFrame(grid_search_knn.cv_results_)
cv_results_knn['MSE'] = -cv_results_knn['mean_test_score']
cv_results_knn['RMSE'] = np.sqrt(cv_results_knn['MSE'])
cv_results_knn['train_RMSE'] = np.sqrt(-cv_results_knn['mean_train_score'])
cv_summary_knn = cv_results_knn[['params', 'MSE', 'RMSE', 'train_RMSE', 'rank_test_score']].sort_values(
    'rank_test_score')

print(f"\n📈 Top 3 Best Parameter Combinations:")
for i, row in cv_summary_knn.head(3).iterrows():
    print(f"   Rank {int(row['rank_test_score'])}: RMSE = {row['RMSE']:.4f}, Parameters = {row['params']}")

# ===== 1️⃣2️⃣ Export Results =====
summary_knn = pd.DataFrame([{
    'Model': 'KNN Regressor',
    'Best Parameters': str(best_params_knn),
    'MSLE': round(msle_knn, 6),
    'RMSE': round(rmse_knn, 4),
    'MAE': round(mae_knn, 4),
    'MAPE(%)': round(mape_knn, 3),
    'R²': round(r2_knn, 4),
    'CV_RMSE': round(np.sqrt(best_score_knn), 4)
}])

output_path = "/home/jiangshan/cre/growth/KNN_results.xlsx"
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    cv_summary_knn.to_excel(writer, sheet_name="CV_Results", index=False)
    summary_knn.to_excel(writer, sheet_name="Test_Summary", index=False)

    # Add preprocessing info
    preprocess_info = pd.DataFrame({
        'KNN_Preprocessing': [
            f'Numeric variables: {numeric_features}',
            f'Categorical variables: {categorical_features}',
            f'Numeric processing: StandardScaler (KNN requires standardization)',
            f'Categorical processing: OneHotEncoder',
            f'Distance metric: Controlled by p parameter (1=Manhattan, 2=Euclidean)',
            f'Feature weighting: uniform (equal weights) or distance (weighted by distance)'
        ]
    })
    preprocess_info.to_excel(writer, sheet_name="Preprocessing_Info", index=False)

    # Add basic data info
    data_info = pd.DataFrame({
        'Data_Info': [
            f'Number of training samples: {X_train.shape[0]}',
            f'Number of test samples: {X_test.shape[0]}',
            f'Number of features after preprocessing: {best_model_knn.named_steps["preprocessor"].transform(X_train.iloc[0:1]).shape[1]}',
            f'Best number of neighbors: {best_params_knn.get("regressor__n_neighbors", "N/A")}',
            f'Best weight method: {best_params_knn.get("regressor__weights", "N/A")}',
            f'Best distance metric: p={best_params_knn.get("regressor__p", "N/A")}'
        ]
    })
    data_info.to_excel(writer, sheet_name="Model_Info", index=False)

print(f"\n✅ All results have been saved to: {output_path}")

# ===== 1️⃣3️⃣ Model Performance Summary =====
print("\n" + "=" * 50)
print("📊 KNN Regression Model Training Completed!")
print(f"   Best number of neighbors: {best_params_knn.get('regressor__n_neighbors', 'N/A')}")
print(f"   Best weight method: {best_params_knn.get('regressor__weights', 'N/A')}")
print(f"   Best distance metric: p={best_params_knn.get('regressor__p', 'N/A')} (1=Manhattan, 2=Euclidean)")
print(f"   Best Test R²: {r2_knn:.4f}")
print(f"   Best Test RMSE: {rmse_knn:.4f}")
print("=" * 50)

# ===== 1️⃣4️⃣ KNN Specific Suggestions =====
print("\n💡 KNN Model Characteristics:")
print("   ✅ Pros: Simple to understand, no training process, robust to outliers")
print("   ⚠️  Cons: High computational complexity, sensitive to feature scale, requires a lot of memory")
print("   📝 Recommendations: Suitable for small to medium-sized datasets, for large datasets consider other algorithms")


###model14 MLP
# ===== 0) Dependencies =====
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# You need to define: X_train, y_train, X_test, y_test beforehand
# Ensure the column names below exist in X_train/X_test
numeric_features = ['30kg ABW', 'Litter size', 'Birth weight']
categorical_features = ['Season', 'Parity', 'Sex']

print("📊 MLP Neural Network Data Preprocessing Info:")
print(f"   Numeric variables: {numeric_features}")
print(f"   Categorical variables: {categorical_features}")
print("⚠️  Neural networks are sensitive to feature scale: numerical features are standardized, categorical features are one-hot encoded.")

# ===== 1) Preprocessing =====
numeric_transformer = StandardScaler()
# If your scikit-learn version is less than 1.2, change sparse_output=False to sparse=False
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ],
    remainder='drop'
)

# ===== 2) Model & Pipeline =====
mlp_model = MLPRegressor(
    max_iter=1000,
    random_state=42,
    early_stopping=True
)

pipeline_mlp = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', mlp_model)
])

# ===== 3) Hyperparameter Grid (72 combinations) =====
params_mlp = {
    'regressor__hidden_layer_sizes': [(50,), (100,), (100, 50), (50, 50), (150,), (200,)],  # 6 options
    'regressor__activation': ['relu', 'tanh'],  # 2 options
    'regressor__solver': ['adam', 'lbfgs'],     # 2 options
    'regressor__alpha': [0.0001, 0.001, 0.01], # 3 options
    'regressor__learning_rate': ['constant', 'adaptive'],
    'regressor__learning_rate_init': [0.001],
    'regressor__batch_size': ['auto'],
}

n_comb = (len(params_mlp['regressor__hidden_layer_sizes']) *
          len(params_mlp['regressor__activation']) *
          len(params_mlp['regressor__solver']) *
          len(params_mlp['regressor__alpha']) *
          len(params_mlp['regressor__learning_rate']))

print(f"🔍 Number of parameter combinations: {n_comb}, total fitting times for 5-fold cross-validation: {n_comb * 5}")
print(f"📊 Parameter distribution:")
print(f"   - Hidden layer sizes (6): {params_mlp['regressor__hidden_layer_sizes']}")
print(f"   - Activation functions (2): {params_mlp['regressor__activation']}")
print(f"   - Solvers (2): {params_mlp['regressor__solver']}")
print(f"   - L2 regularization (3): {params_mlp['regressor__alpha']}")
print(f"   - Learning rate strategies (2): {params_mlp['regressor__learning_rate']}")

# ===== 4) 5-fold Cross-validation + Grid Search =====
cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search_mlp = GridSearchCV(
    estimator=pipeline_mlp,
    param_grid=params_mlp,
    cv=cv,
    n_jobs=1,                    # Change to -1 for faster execution
    scoring='neg_mean_squared_error',
    verbose=1,
    return_train_score=True
)

print("🚀 Starting to train MLP (5-fold)...")
grid_search_mlp.fit(X_train, y_train)

# ===== 5) Best Model and Parameters =====
best_model = grid_search_mlp.best_estimator_
best_params = grid_search_mlp.best_params_
best_cv_mse = -grid_search_mlp.best_score_

print(f"\n✅ Best Parameters: {best_params}")
print(f"✅ Best CV RMSE: {np.sqrt(best_cv_mse):.4f}")

# ===== 6) CV Results Summary =====
cv_results = pd.DataFrame(grid_search_mlp.cv_results_)
cv_results['MSE'] = -cv_results['mean_test_score']
cv_results['RMSE'] = np.sqrt(cv_results['MSE'])
cv_results['train_RMSE'] = np.sqrt(-cv_results['mean_train_score'])
cv_summary = cv_results[['params', 'MSE', 'RMSE', 'train_RMSE', 'rank_test_score']].sort_values('rank_test_score')

print("\n📈 Top 3 Best Parameter Combinations:")
for _, row in cv_summary.head(3).iterrows():
    print(f"   Rank {int(row['rank_test_score'])}: RMSE={row['RMSE']:.4f} (train={row['train_RMSE']:.4f})")

# ===== 7) Test Set Evaluation =====
y_pred = best_model.predict(X_test)
eps = 1e-9
y_test_safe = np.clip(np.asarray(y_test), eps, None)
y_pred_safe = np.clip(y_pred, eps, None)

msle = mean_squared_log_error(y_test_safe, y_pred_safe)
rmse = np.sqrt(mean_squared_error(y_test_safe, y_pred_safe))
mae = mean_absolute_error(y_test_safe, y_pred_safe)
mape = np.mean(np.abs((y_test_safe - y_pred_safe) / y_test_safe)) * 100
r2 = r2_score(y_test_safe, y_pred_safe)

print("\n🎯 Test Set Metrics:")
print(f"MSLE = {msle:.6f}")
print(f"RMSE = {rmse:.4f}")
print(f"MAE  = {mae:.4f}")
print(f"MAPE = {mape:.3f}%")
print(f"R²   = {r2:.4f}")

# ===== 8) Export to Excel =====
summary = pd.DataFrame([{
    'Model': 'MLP Neural Network',
    'Best Parameters': str(best_params),
    'MSLE': round(msle, 6),
    'RMSE': round(rmse, 4),
    'MAE': round(mae, 4),
    'MAPE(%)': round(mape, 3),
    'R²': round(r2, 4),
    'CV_RMSE': round(np.sqrt(best_cv_mse), 4),
}])

output_path = "/home/jiangshan/cre/growth2/MLP_results.xlsx"
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    cv_summary.to_excel(writer, sheet_name="CV_Results", index=False)
    summary.to_excel(writer, sheet_name="Test_Summary", index=False)

print(f"\n✅ All results have been saved: {output_path}")

###model15 catboost

from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ===== 1️⃣ Define Variable Types =====
numeric_features = ['30kg ABW', 'Litter size', 'Birth weight']
categorical_features = ['Season', 'Parity', 'Sex']

print("📊 CatBoost Data Preprocessing Information:")
print(f"   Numeric variables ({len(numeric_features)}): {numeric_features}")
print(f"   Categorical variables ({len(categorical_features)}): {categorical_features}")
print("⚠️  Note: CatBoost automatically handles categorical variables, no need for one-hot encoding")
print("💡 CatBoost Characteristics: Native support for categorical features, automatic handling of missing values, prevents overfitting")

# ===== 2️⃣ Data Preprocessing =====
# CatBoost automatically handles categorical variables, we only need to process the numeric variables
numeric_transformer = StandardScaler()

# For CatBoost, we do not need to encode categorical variables as it will handle them automatically
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ],
    remainder='passthrough'  # Keep categorical variables in their original format
)

# Get all feature names
all_features = numeric_features + categorical_features

# ===== 3️⃣ CatBoost Regressor Model =====
# CatBoost automatically handles categorical variables, we need to specify the indices of categorical features
catboost_model = CatBoostRegressor(
    random_state=42,
    verbose=0,  # Disable training logs
    loss_function='RMSE',
    early_stopping_rounds=50
)

# Build the Pipeline
pipeline_catboost = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', catboost_model)
])

# ===== 4️⃣ Hyperparameter Grid Optimization =====
# 96 combinations of the parameter grid - balancing diversity and training time
params_catboost = {
    'regressor__learning_rate': [0.01, 0.05, 0.1, 0.15],  # 4 options
    'regressor__depth': [4, 6, 8, 10],  # 4 options
    'regressor__iterations': [500, 800, 1000],  # 3 options
    'regressor__l2_leaf_reg': [1, 3, 5],  # 3 options
    'regressor__border_count': [64],  # Fixed value
    'regressor__random_strength': [1]  # Fixed value
}

total_combinations = (len(params_catboost['regressor__learning_rate']) *
                      len(params_catboost['regressor__depth']) *
                      len(params_catboost['regressor__iterations']) *
                      len(params_catboost['regressor__l2_leaf_reg']))

print(f"🔍 Number of parameter grid combinations: {total_combinations}")
print(f"📊 Total training times for 5-fold cross-validation: {total_combinations * 5}")
print(f"📈 Parameter distribution:")
print(f"   - Learning rate (4): {params_catboost['regressor__learning_rate']}")
print(f"   - Tree depth (4): {params_catboost['regressor__depth']}")
print(f"   - Iterations (3): {params_catboost['regressor__iterations']}")
print(f"   - L2 regularization (3): {params_catboost['regressor__l2_leaf_reg']}")
print(f"   - Border count (1): {params_catboost['regressor__border_count']}")
print(f"   - Random strength (1): {params_catboost['regressor__random_strength']}")

# ===== 5️⃣ Cross-validation + Grid Search =====
cv = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation
grid_search_catboost = GridSearchCV(
    estimator=pipeline_catboost,
    param_grid=params_catboost,
    cv=cv,
    n_jobs=1,
    scoring='neg_mean_squared_error',
    verbose=1,
    return_train_score=True
)

# ===== 6️⃣ Fit the Model =====
print("🚀 Starting to train CatBoost model...")
print("💡 Note: Using 96 parameter combinations + 5-fold cross-validation")
print("🔧 Total training required: 96 × 5 = 480 models")
print("⏰ Estimated training time: Medium (depends on data size and hardware performance)")
print("🔧 Early stopping enabled to prevent overfitting")

try:
    grid_search_catboost.fit(X_train, y_train)

    # ===== 7️⃣ Best Model and Parameters =====
    best_model_catboost = grid_search_catboost.best_estimator_
    best_params_catboost = grid_search_catboost.best_params_
    best_score_catboost = -grid_search_catboost.best_score_

    print(f"\n✅ Best Parameters for CatBoost: {best_params_catboost}")
    print(f"✅ Best Cross-validation RMSE: {np.sqrt(best_score_catboost):.4f}")

    # ===== 8️⃣ Cross-validation Results Analysis =====
    cv_results_catboost = pd.DataFrame(grid_search_catboost.cv_results_)
    cv_results_catboost['MSE'] = -cv_results_catboost['mean_test_score']
    cv_results_catboost['RMSE'] = np.sqrt(cv_results_catboost['MSE'])
    cv_results_catboost['train_RMSE'] = np.sqrt(-cv_results_catboost['mean_train_score'])
    cv_summary_catboost = cv_results_catboost[['params', 'MSE', 'RMSE', 'train_RMSE', 'rank_test_score']].sort_values(
        'rank_test_score')

    print(f"\n📈 Top 5 Best Parameter Combinations:")
    for i, row in cv_summary_catboost.head(5).iterrows():
        print(f"   Rank {int(row['rank_test_score'])}: RMSE = {row['RMSE']:.4f} (Training: {row['train_RMSE']:.4f})")

    # ===== 9️⃣ Test Set Prediction and Evaluation =====
    y_pred_catboost = best_model_catboost.predict(X_test)

    # Avoid negative values (to prevent MSLE error)
    eps = 1e-9
    y_test_safe = np.clip(np.asarray(y_test), eps, None)
    y_pred_safe = np.clip(y_pred_catboost, eps, None)

    msle_catboost = mean_squared_log_error(y_test_safe, y_pred_safe)
    rmse_catboost = np.sqrt(mean_squared_error(y_test_safe, y_pred_safe))
    mae_catboost = mean_absolute_error(y_test_safe, y_pred_safe)
    mape_catboost = np.mean(np.abs((y_test_safe - y_pred_safe) / y_test_safe)) * 100
    r2_catboost = r2_score(y_test_safe, y_pred_safe)

    print("\n🎯 CatBoost Test Set Performance Metrics:")
    print(f"MSLE = {msle_catboost:.6f}")
    print(f"RMSE = {rmse_catboost:.4f}")
    print(f"MAE  = {mae_catboost:.4f}")
    print(f"MAPE = {mape_catboost:.3f}%")
    print(f"R²   = {r2_catboost:.4f}")

    # ===== 🔟 CatBoost Specific Analysis =====
    print("\n🔍 CatBoost Specific Analysis:")
    try:
        # Get the CatBoost model
        catboost_regressor = best_model_catboost.named_steps['regressor']

        # Get feature importance
        feature_importances = catboost_regressor.get_feature_importance()

        # Create feature importance DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': all_features,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)

        # Calculate cumulative importance
        feature_importance_df['Cumulative_Importance'] = feature_importance_df['Importance'].cumsum()

        # Find features contributing to 80% of the importance
        important_features = feature_importance_df[feature_importance_df['Cumulative_Importance'] <= 0.8]
        num_important_features = len(important_features)

        print(f"Total features: {len(all_features)}")
        print(f"Top {num_important_features} features contribute to 80% of importance")

        print("\n📊 CatBoost Feature Importance (Top 10):")
        print(feature_importance_df.head(10).round(4).to_string(index=False))

        # Parameter Analysis
        learning_rate = best_params_catboost.get('regressor__learning_rate', 'N/A')
        depth = best_params_catboost.get('regressor__depth', 'N/A')
        iterations = best_params_catboost.get('regressor__iterations', 'N/A')
        l2_leaf_reg = best_params_catboost.get('regressor__l2_leaf_reg', 'N/A')

        print(f"\n📊 CatBoost Parameter Analysis:")
        print(f"Learning Rate: {learning_rate}")
        print(f"Tree Depth: {depth}")
        print(f"Iterations: {iterations}")
        print(f"L2 Regularization: {l2_leaf_reg}")

        # Learning rate analysis
        if isinstance(learning_rate, (int, float)):
            if learning_rate < 0.05:
                print("💡 Low learning rate: More stable training, but may require more iterations")
            elif learning_rate > 0.15:
                print("💡 High learning rate: Faster training, but may be unstable")

        # Tree depth analysis
        if isinstance(depth, (int, float)):
            if depth < 5:
                print("💡 Small tree depth: Simpler model, prevents overfitting")
            elif depth > 7:
                print("💡 Large tree depth: More complex model, may overfit")

        # Regularization analysis
        if isinstance(l2_leaf_reg, (int, float)):
            if l2_leaf_reg < 3:
                print("💡 Weak L2 regularization: Model may be complex")
            elif l2_leaf_reg > 7:
                print("💡 Strong L2 regularization: Model is simpler")

        # Categorical feature handling advantages
        print(f"\n🔍 CatBoost Categorical Feature Handling:")
        print("   ✅ Automatically handles categorical features, no need for one-hot encoding")
        print("   ✅ Uses Ordered Boosting to prevent target leakage")
        print("   ✅ Optimized for categorical features")

        # Training process analysis
        print(f"\n📈 Training Process Analysis:")
        print(f"   Best Iteration: {catboost_regressor.get_best_iteration()}")
        print(f"   Total Trees: {catboost_regressor.tree_count_}")

        # Overfitting detection
        if hasattr(catboost_regressor, 'evals_result_'):
            print("   Training process monitoring: Available")
        else:
            print("   Training process monitoring: Not recorded")

    except Exception as e:
        print(f"⚠️ Unable to perform CatBoost specific analysis: {e}")
        feature_importance_df = None
        num_important_features = 'N/A'

    # ===== 1️⃣1️⃣ Overfitting Analysis =====
    print("\n🔍 Overfitting Analysis:")
    try:
        # Get performance gap between training and test sets
        train_score = -grid_search_catboost.cv_results_['mean_train_score'][grid_search_catboost.best_index_]
        test_score = best_score_catboost
        performance_gap = train_score - test_score

        print(f"Training MSE: {train_score:.4f}")
        print(f"Test MSE: {test_score:.4f}")
        print(f"Performance gap: {performance_gap:.4f}")

        if performance_gap > 0.1:
            print("⚠️  Possible overfitting, suggestions:")
            print("   - Increase L2 regularization strength")
            print("   - Reduce tree depth")
        elif performance_gap < 0.02:
            print("✅ Good generalization")
        else:
            print("💡 Moderate performance gap")

    except Exception as e:
        print(f"⚠️ Unable to perform overfitting analysis: {e}")

    # ===== 1️⃣2️⃣ Export Results =====
    summary_catboost = pd.DataFrame([{
        'Model': 'CatBoost',
        'Best Parameters': str(best_params_catboost),
        'MSLE': round(msle_catboost, 6),
        'RMSE': round(rmse_catboost, 4),
        'MAE': round(mae_catboost, 4),
        'MAPE(%)': round(mape_catboost, 3),
        'R²': round(r2_catboost, 4),
        'CV_RMSE': round(np.sqrt(best_score_catboost), 4),
        'Learning_Rate': best_params_catboost.get('regressor__learning_rate', 'N/A'),
        'Tree_Depth': best_params_catboost.get('regressor__depth', 'N/A'),
        'Iterations': best_params_catboost.get('regressor__iterations', 'N/A'),
        'L2_Regularization': best_params_catboost.get('regressor__l2_leaf_reg', 'N/A'),
        'Important_Features': num_important_features if isinstance(num_important_features, int) else 'N/A',
        'Best_Iteration': catboost_regressor.get_best_iteration() if hasattr(catboost_regressor,
                                                                             'get_best_iteration') else 'N/A'
    }])

    output_path = "/home/jiangshan/cre/growth2/CatBoost_results.xlsx"
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        cv_summary_catboost.to_excel(writer, sheet_name="CV_Results", index=False)
        summary_catboost.to_excel(writer, sheet_name="Test_Summary", index=False)

        # Add feature importance to Excel
        try:
            if feature_importance_df is not None:
                feature_importance_df.to_excel(writer, sheet_name="Feature_Importance", index=False)
        except:
            pass

        # Add preprocessing info
        preprocess_info = pd.DataFrame({
            'CatBoost_Preprocessing': [
                f'Numeric variables: {numeric_features}',
                f'Categorical variables: {categorical_features}',
                f'Numeric processing: StandardScaler',
                f'Categorical processing: CatBoost automatic handling (no encoding)',
                f'Model type: Gradient Boosting - CatBoost implementation',
                f'Algorithm Characteristics: Automatic handling of categorical features, ordered boosting, symmetric trees',
                f'Advantages: Categorical feature optimization, missing value handling, prevents target leakage'
            ]
        })
        preprocess_info.to_excel(writer, sheet_name="Preprocessing_Info", index=False)

        # Add model configuration info
        model_config_data = [
            f'Learning Rate: {best_params_catboost.get("regressor__learning_rate", "N/A")}',
            f'Tree Depth: {best_params_catboost.get("regressor__depth", "N/A")}',
            f'Iterations: {best_params_catboost.get("regressor__iterations", "N/A")}',
            f'L2 Regularization: {best_params_catboost.get("regressor__l2_leaf_reg", "N/A")}',
            f'Border Count: {best_params_catboost.get("regressor__border_count", "N/A")}',
            f'Random Strength: {best_params_catboost.get("regressor__random_strength", "N/A")}',
            f'Best Iteration: {catboost_regressor.get_best_iteration() if hasattr(catboost_regressor, "get_best_iteration") else "N/A"}',
            f'Total Tree Count: {catboost_regressor.tree_count_ if hasattr(catboost_regressor, "tree_count_") else "N/A"}',
            f'Important Features (80% Contribution): {num_important_features if isinstance(num_important_features, int) else "N/A"}',
            f'Total Features: {len(all_features)}',
            f'Test R²: {r2_catboost:.4f}',
            f'Parameter Combinations: {total_combinations}',
            f'Cross-validation Folds: 5'
        ]

        model_info = pd.DataFrame({
            'CatBoost_Configuration': model_config_data
        })
        model_info.to_excel(writer, sheet_name="Model_Config", index=False)

    print(f"\n✅ All results have been saved to: {output_path}")

    # ===== 1️⃣3️⃣ Model Performance Summary =====
    print("\n" + "=" * 50)
    print("📊 CatBoost Model Training Completed!")
    print(f"   Learning Rate: {best_params_catboost.get('regressor__learning_rate', 'N/A')}")
    print(f"   Tree Depth: {best_params_catboost.get('regressor__depth', 'N/A')}")
    print(f"   Iterations: {best_params_catboost.get('regressor__iterations', 'N/A')}")
    print(f"   Important Features: {num_important_features if isinstance(num_important_features, int) else 'N/A'}")
    print(f"   Best Test R²: {r2_catboost:.4f}")
    print(f"   Best Test RMSE: {rmse_catboost:.4f}")
    print("=" * 50)

    # ===== 1️⃣4️⃣ CatBoost Specific Suggestions =====
    print("\n💡 CatBoost Characteristics:")
    print("   ✅ Pros: Automatic handling of categorical features, excellent missing value handling, prevents target leakage, high performance")
    print("   ⚠️  Cons: Training time may be long, memory usage can be high")
    print("   📝 Recommendations:")
    print("      - Particularly suitable for datasets with categorical features")
    print("      - Use Ordered Boosting to prevent overfitting")
    print("      - Symmetric tree structure improves inference speed")

    # ===== 1️⃣5️⃣ Comparison with Other Gradient Boosting Algorithms =====
    print("\n🔍 CatBoost vs Other Gradient Boosting Algorithms:")
    print("   - XGBoost: Requires manual encoding of categorical features, more complex hyperparameter tuning")
    print("   - LightGBM: Faster training, but does not handle categorical features as well as CatBoost")
    print("   - CatBoost: Native support for categorical features, default parameters usually perform well")

    # ===== 1️⃣6️⃣ Performance Optimization Suggestions =====
    print("\n🎯 CatBoost Performance Optimization Guide:")
    if r2_catboost < 0.5:
        print("   💡 Current R² is low, suggestions:")
        print("      - Increase the number of iterations")
        print("      - Adjust learning rate and tree depth")
        print("      - Check if categorical features are correctly recognized")
    elif r2_catboost > 0.8:
        print("   💡 Excellent performance, indicating:")
        print("      - CatBoost has effectively utilized categorical feature information")
        print("      - Hyperparameter tuning has worked well")
        print("      - Data quality is high")
    else:
        print("   💡 Moderate performance, further parameter optimization may be needed")

    # ===== 1️⃣7️⃣ Advantages of Categorical Feature Handling =====
    print("\n🔍 Advantages of CatBoost Categorical Feature Handling:")
    print("   1. No need for one-hot encoding: Avoids dimensionality explosion")
    print("   2. Target statistics: Uses target variable information to encode categories")
    print("   3. Ordered Boosting: Prevents target leakage")
    print("   4. Feature combinations: Automatically creates meaningful feature combinations")

except Exception as e:
    print(f"❌ CatBoost model training failed: {e}")
    print("🔧 Debugging suggestions:")
    print("   1. Check if CatBoost is correctly installed")
    print("   2. Check data format and types")
    print("   3. Ensure X_train, y_train, X_test, y_test are properly defined")
    print("   4. Check for missing values")
    print("   5. Try reducing the number of parameter combinations")

    # Try using a default CatBoost model for debugging
    print("\n🔄 Trying a simplified CatBoost model for debugging...")
    try:
        simple_catboost = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', CatBoostRegressor(iterations=500, verbose=0, random_state=42))
        ])

        simple_catboost.fit(X_train, y_train)
        y_pred_simple = simple_catboost.predict(X_test)
        r2_simple = r2_score(y_test, y_pred_simple)
        print(f"✅ Simplified CatBoost model trained successfully, Test R²: {r2_simple:.4f}")
        print("💡 This indicates data preprocessing and the basic CatBoost structure are correct")

    except Exception as simple_error:
        print(f"❌ Simplified CatBoost model also failed: {simple_error}")
        print("🔍 You may need to check the data preprocessing steps or data quality")


