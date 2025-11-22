from sklearn import metrics
import numpy as np

print("📊 Calculating CatBoost model performance metrics for training and testing datasets...")

# True values
y_train_true = y_train.values
y_test_true = y_test.values

# Predict using CatBoost model
y_pred_train = catboost.predict(X_train)
y_pred_test = catboost.predict(X_test)

# Training set metrics
mse_train = metrics.mean_squared_error(y_train_true, y_pred_train)
rmse_train = np.sqrt(mse_train)
mae_train = metrics.mean_absolute_error(y_train_true, y_pred_train)
r2_train = metrics.r2_score(y_train_true, y_pred_train)

# Test set metrics
mse_test = metrics.mean_squared_error(y_test_true, y_pred_test)
rmse_test = np.sqrt(mse_test)
mae_test = metrics.mean_absolute_error(y_test_true, y_pred_test)
r2_test = metrics.r2_score(y_test_true, y_pred_test)

# Calculate MAPE (Mean Absolute Percentage Error)
def calculate_mape(y_true, y_pred):
    eps = 1e-9
    y_true_safe = np.clip(np.asarray(y_true), eps, None)
    y_pred_safe = np.clip(y_pred, eps, None)
    return np.mean(np.abs((y_true_safe - y_pred_safe) / y_true_safe)) * 100

mape_train = calculate_mape(y_train_true, y_pred_train)
mape_test = calculate_mape(y_test_true, y_pred_test)

# Output results
print("=" * 50)
print("🏋️‍♂️ Training set performance metrics:")
print("=" * 50)
print(f"Mean Squared Error (MSE): {mse_train:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_train:.4f}")
print(f"Mean Absolute Error (MAE): {mae_train:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape_train:.2f}%")
print(f"R²: {r2_train:.4f}")

print("\n" + "=" * 50)
print("🧪 Test set performance metrics:")
print("=" * 50)
print(f"Mean Squared Error (MSE): {mse_test:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_test:.4f}")
print(f"Mean Absolute Error (MAE): {mae_test:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape_test:.2f}%")
print(f"R²: {r2_test:.4f}")

# Calculate overfitting degree
overfitting_rmse = rmse_train - rmse_test
overfitting_r2 = r2_test - r2_train

print("\n" + "=" * 50)
print("📈 Overfitting Analysis:")
print("=" * 50)
print(f"RMSE difference (Training - Test): {overfitting_rmse:.4f}")
print(f"R² difference (Test - Training): {overfitting_r2:.4f}")

if overfitting_rmse < 0.1 and abs(overfitting_r2) < 0.1:
    print("✅ The model generalizes well, low overfitting")
elif overfitting_rmse > 0.2 or abs(overfitting_r2) > 0.15:
    print("⚠️ Possible overfitting detected")
else:
    print("🔍 The model is performing normally")

# Feature importance (keeping consistent with the original code)
print("\n" + "=" * 50)
print("📊 Feature Importance:")
print("=" * 50)
feature_importance = catboost.get_feature_importance()
feature_names = X_train.columns

# Create DataFrame for feature importance and sort
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

for _, row in importance_df.iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

# Model configuration (keeping consistent with the original code)
print("\n" + "=" * 50)
print("⚙️  Model Configuration:")
print("=" * 50)
print(f"  Iterations: 1000")
print(f"  Learning rate: 0.01")
print(f"  Tree depth: 8")
print(f"  L2 regularization: 1")
print(f"  Border count: 64")
print(f"  Random strength: 1")
print(f"  Random seed: 8")

# Optional: Plot the scatter plot of predictions vs true values
print("\n" + "=" * 50)
print("📊 Generating prediction result visualization...")
print("=" * 50)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Training set scatter plot
ax1.scatter(y_train_true, y_pred_train, alpha=0.6, s=20)
ax1.plot([y_train_true.min(), y_train_true.max()], [y_train_true.min(), y_train_true.max()], 'r--', lw=2)
ax1.set_xlabel('True Values')
ax1.set_ylabel('Predicted Values')
ax1.set_title(f'Training Set: R² = {r2_train:.4f}')

# Test set scatter plot
ax2.scatter(y_test_true, y_pred_test, alpha=0.6, s=20)
ax2.plot([y_test_true.min(), y_test_true.max()], [y_test_true.min(), y_test_true.max()], 'r--', lw=2)
ax2.set_xlabel('True Values')
ax2.set_ylabel('Predicted Values')
ax2.set_title(f'Test Set: R² = {r2_test:.4f}')

plt.tight_layout()
plt.savefig("CatBoost_Performance_Scatter.pdf", format='pdf', bbox_inches='tight', dpi=1200)
plt.show()
print("✅ The performance scatter plot has been saved as CatBoost_Performance_Scatter.pdf")


import scipy.stats as stats
# scale_factor is used to scale the confidence interval
scale_factor = 1.5  # Adjust this value, the larger the value, the wider the confidence interval. scale_factor = 1 is the theoretical standard confidence interval width
confidence = 0.95  # 95% confidence level
# Fit the training set line
z_train = np.polyfit(y_train, y_pred_train, 1)
p_train = np.poly1d(z_train)
predicted_values_train = p_train(y_train)
residuals_train = y_pred_train - predicted_values_train
mean_error_train = np.mean(residuals_train**2)
t_value_train = stats.t.ppf((1 + confidence) / 2., len(y_train) - 1)
ci_train = t_value_train * scale_factor * np.sqrt(mean_error_train) * np.sqrt(1 / len(y_train) + (y_train - np.mean(y_train))**2 / np.sum((y_train - np.mean(y_train))**2))
x_extended_train = np.linspace(min(y_train), max(y_train), 100)
predicted_extended_train = p_train(x_extended_train)
ci_extended_train = t_value_train * scale_factor * np.sqrt(mean_error_train) * np.sqrt(1 / len(y_train) + (x_extended_train - np.mean(y_train))**2 / np.sum((y_train - np.mean(y_train))**2))

# Fit the test set line
z_test = np.polyfit(y_test, y_pred_test, 1)
p_test = np.poly1d(z_test)
predicted_values_test = p_test(y_test)
residuals_test = y_pred_test - predicted_values_test
mean_error_test = np.mean(residuals_test**2)
t_value_test = stats.t.ppf((1 + confidence) / 2., len(y_test) - 1)
ci_test = t_value_test * scale_factor * np.sqrt(mean_error_test) * np.sqrt(1 / len(y_test) + (y_test - np.mean(y_test))**2 / np.sum((y_test - np.mean(y_test))**2))
x_extended_test = np.linspace(min(y_test), max(y_test), 100)
predicted_extended_test = p_test(x_extended_test)
ci_extended_test = t_value_test * scale_factor * np.sqrt(mean_error_test) * np.sqrt(1 / len(y_test) + (x_extended_test - np.mean(y_test))**2 / np.sum((y_test - np.mean(y_test))**2))


# Set new color scheme
train_color = '#1f77b4'  # Training set main color: Blue
test_color = '#ff7f0e'   # Test set main color: Orange
confidence_train_color = '#aec7e8'  # Training set confidence interval light blue
confidence_test_color = '#ffbb78'   # Test set confidence interval light orange

# Set figure size and layout
fig = plt.figure(figsize=(10, 8), dpi=1200)
gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
ax_main = fig.add_subplot(gs[1:, :-1])  # Main plot
ax_hist_x = fig.add_subplot(gs[0, :-1], sharex=ax_main)  # Histogram above
ax_hist_y = fig.add_subplot(gs[1:, -1], sharey=ax_main)  # Histogram on the right

# Plot the training set
ax_main.scatter(y_train, y_pred_train, color=train_color, label="Training Predicted Values", alpha=0.6)
ax_main.plot(y_train, p_train(y_train), color=train_color, alpha=0.9, label=f"Training Line of Best Fit\n$R^2$ = {r2_train:.2f}, MAE = {mae_train:.2f}")
ax_main.fill_between(x_extended_train, predicted_extended_train - ci_extended_train, predicted_extended_train + ci_extended_train,
                     color=confidence_train_color, alpha=0.5, label="Training 95% Confidence Interval")

# Plot the test set
ax_main.scatter(y_test, y_pred_test, color=test_color, label="Testing Predicted Values", alpha=0.6)
ax_main.plot(y_test, p_test(y_test), color=test_color, alpha=0.9, label=f"Testing Line of Best Fit\n$R^2$ = {r2_test:.2f}, MAE = {mae_test:.2f}")
ax_main.fill_between(x_extended_test, predicted_extended_test - ci_extended_test, predicted_extended_test + ci_extended_test,
                     color=confidence_test_color, alpha=0.5, label="Testing 95% Confidence Interval")

# Add reference line
ax_main.plot([min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
             [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
             color='grey', linestyle='--', alpha=0.6, label="1:1 Line")

# Set main plot
ax_main.set_xlabel("Observed Values", fontsize=12)
ax_main.set_ylabel("Predicted Values", fontsize=12)
ax_main.legend(loc="upper left", fontsize=10)

# Plot histogram above (distribution of observed values)
ax_hist_x.hist(y_train, bins=20, color=train_color, alpha=0.7, edgecolor='black', label="Training Observed Distribution")
ax_hist_x.hist(y_test, bins=20, color=test_color, alpha=0.7, edgecolor='black')
ax_hist_x.tick_params(labelbottom=False)  # Hide x-axis labels

# Plot histogram on the right (distribution of predicted values)
ax_hist_y.hist(y_pred_train, bins=20, orientation='horizontal', color=train_color, alpha=0.7, edgecolor='black')
ax_hist_y.hist(y_pred_test, bins=20, orientation='horizontal', color=test_color, alpha=0.7, edgecolor='black')
ax_hist_y.set_xlabel("Frequency", fontsize=12)
ax_hist_y.tick_params(labelleft=False)  # Hide y-axis labels

# Save and display the image
plt.savefig('train_test_combined_with_histograms_and_confidence_intervals_new_colors.pdf', format='pdf', bbox_inches='tight')
plt.show()
print("✅ Performance scatter plot saved as CatBoost_Performance_Scatter.pdf")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# =========================
# Global font settings - Times New Roman
# =========================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42  # Ensure text is editable in PDF
plt.rcParams['ps.fonttype'] = 42   # Ensure text is editable in PostScript

# =========================
# 1. Read external validation dataset, create external cohort
# =========================

ext_path = "/home/jiangshan/cre/growth2/growth888.xlsx"
df_ext = pd.read_excel(ext_path)

# Features are consistent with those in the internal training set
features = ['30kg ABW', 'Litter size', 'Season', 'Birth weight', 'Parity', 'Sex']

X_ext = df_ext[features]
y_ext = df_ext['Y']

print(f"External dataset size: X_ext = {X_ext.shape}, y_ext = {y_ext.shape}")

# =========================
# 2. Define a function to calculate R²
# =========================
def get_r2(model, X, y, idx=None):
    """
    Given the model, features X, and labels y (optional: index idx), return the R² for that subset.
    """
    if idx is not None:
        X = X.iloc[idx]
        y = y.iloc[idx]
    y_pred = model.predict(X)
    return r2_score(y, y_pred)

# Internal cohort uses your previous test set X_test, y_test
r2_internal = get_r2(catboost, X_test, y_test)
r2_external = get_r2(catboost, X_ext, y_ext)
delta_r2 = r2_external - r2_internal  # ΔR² = External - Internal

print("===== Baseline R² on two cohorts =====")
print(f"Internal R² (test set): {r2_internal:.4f}")
print(f"External R²:            {r2_external:.4f}")
print(f"ΔR² (external - internal): {delta_r2:.4f}")

# =========================
# 3. Non-parametric bootstrap evaluation of ΔR²
# =========================

n_boot = 1000        # Number of bootstrap iterations
rng = np.random.default_rng(42)  # Random seed to ensure reproducibility

# internal cohort = test set
n_int = len(y_test)
n_ext = len(y_ext)

r2_int_boot = np.empty(n_boot)
r2_ext_boot = np.empty(n_boot)
delta_boot = np.empty(n_boot)

for b in range(n_boot):
    # Independently resample (with replacement) internal/external cohorts
    idx_int = rng.integers(0, n_int, size=n_int)
    idx_ext = rng.integers(0, n_ext, size=n_ext)

    r2_i = get_r2(catboost, X_test, y_test, idx=idx_int)
    r2_e = get_r2(catboost, X_ext, y_ext, idx=idx_ext)

    r2_int_boot[b] = r2_i
    r2_ext_boot[b] = r2_e
    delta_boot[b] = r2_e - r2_i  # ΔR² = external - internal

# As per your method description: P-value = Proportion of external R² <= internal R²
p_value = np.mean(r2_ext_boot <= r2_int_boot)

# Also give the 95% bootstrap confidence interval for ΔR²
alpha = 0.05
ci_lower, ci_upper = np.percentile(delta_boot, [100*alpha/2, 100*(1 - alpha/2)])

print("\n===== Bootstrap results (ΔR² = R²_ext - R²_int) =====")
print(f"Observed ΔR²: {delta_r2:.4f}")
print(f"Bootstrap mean ΔR²: {delta_boot.mean():.4f}")
print(f"Bootstrap 95% CI for ΔR²: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"P-value (Pr[R²_ext ≤ R²_int]): {p_value:.4f}")

# =========================
# 4. Performance metrics calculation function
# =========================
def calculate_mape(y_true, y_pred):
    eps = 1e-9  # To prevent division by zero
    y_true_safe = np.clip(np.asarray(y_true), eps, None)
    y_pred_safe = np.clip(np.asarray(y_pred), eps, None)
    return np.mean(np.abs((y_true_safe - y_pred_safe) / y_true_safe)) * 100

# Internal test set metrics (internal cohort)
y_pred_int = catboost.predict(X_test)
rmse_int = np.sqrt(mean_squared_error(y_test, y_pred_int))
mae_int = mean_absolute_error(y_test, y_pred_int)
mape_int = calculate_mape(y_test, y_pred_int)

# External independent test set metrics (external cohort, growth33.xlsx)
y_pred_ext = catboost.predict(X_ext)
rmse_ext = np.sqrt(mean_squared_error(y_ext, y_pred_ext))
mae_ext = mean_absolute_error(y_ext, y_pred_ext)
mape_ext = calculate_mape(y_ext, y_pred_ext)

print("===== Internal test set (X_test, y_test) =====")
print(f"RMSE: {rmse_int:.2f}")
print(f"MAE:  {mae_int:.2f}")
print(f"MAPE: {mape_int:.2f}%")

print("\n===== External validation set (growth33.xlsx) =====")
print(f"RMSE: {rmse_ext:.2f}")
print(f"MAE:  {mae_ext:.2f}")
print(f"MAPE: {mape_ext:.2f}%")

# =========================
# 5. Plotting the bootstrap distribution of ΔR² (including P-value)
# =========================
# Create figure and set font
fig, ax = plt.subplots(figsize=(8, 6))

# Plot histogram
n, bins, patches = ax.hist(delta_boot, bins=30, edgecolor='black', alpha=0.7,
                           color='skyblue', density=False)

# Add observed value line
ax.axvline(delta_r2, color='red', linestyle='--', linewidth=2.5,
           label=f'Observed ΔR² = {delta_r2:.4f}')

# Add zero line (no difference reference line)
ax.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)

# Set labels and title (using Times New Roman font)
ax.set_xlabel("ΔR² (R²_external - R²_internal)", fontsize=14, weight='bold', fontfamily='Times New Roman')
ax.set_ylabel("Frequency", fontsize=14, weight='bold', fontfamily='Times New Roman')
ax.set_title("Bootstrap Distribution of ΔR²\nExternal vs Internal Validation",
             fontsize=16, weight='bold', pad=20, fontfamily='Times New Roman')

# Add P-value and statistics text in the plot
stats_text = (
    f"Observed ΔR² = {delta_r2:.4f}\n"
    f"Bootstrap Mean ΔR² = {delta_boot.mean():.4f}\n"
    f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]\n"
    f"P-value = {p_value:.4f}"
)

# Based on P-value significance, choose text box color
if p_value < 0.001:
    box_color = 'lightcoral'
    significance_star = "***"
elif p_value < 0.01:
    box_color = 'lightcoral'
    significance_star = "**"
elif p_value < 0.05:
    box_color = 'lightcoral'
    significance_star = "*"
else:
    box_color = 'lightgreen'
    significance_star = "ns"

# Add statistics text box (using Times New Roman font)
ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=11, fontfamily='Times New Roman',
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))

# Set legend (using Times New Roman font)
ax.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True, framealpha=0.9, prop={'family': 'Times New Roman'})

# Set grid
ax.grid(True, alpha=0.3)

# Save as vector PDF (ensure text is editable)
plt.tight_layout()
plt.savefig("bootstrap_delta_R2_distribution_with_pvalue.pdf",
            format='pdf',
            bbox_inches='tight',
            dpi=300,
            transparent=False,
            metadata={'Creator': 'Python Matplotlib', 'Title': 'Bootstrap ΔR² Distribution'})
plt.show()

print("✅ ΔR² bootstrap distribution plot saved as bootstrap_delta_R2_distribution_with_pvalue.pdf")

# =========================
# 6. Print statistical significance interpretation
# =========================
print("\n" + "="*50)
print("📊 STATISTICAL SIGNIFICANCE INTERPRETATION")
print("="*50)

if p_value < 0.001:
    print(f"🔴 HIGHLY SIGNIFICANT (P < 0.001)")
    print("   The model performance differs significantly between")
    print("   internal and external validation cohorts.")
elif p_value < 0.01:
    print(f"🔴 VERY SIGNIFICANT (P < 0.01)")
    print("   Strong evidence of performance difference between cohorts.")
elif p_value < 0.05:
    print(f"🟡 SIGNIFICANT (P < 0.05)")
    print("   Evidence of performance difference exists.")
else:
    print(f"🟢 NOT SIGNIFICANT (P ≥ 0.05)")
    print("   No strong evidence of performance difference between cohorts.")

print(f"\nInterpretation:")
if delta_r2 > 0:
    print(f"   External cohort shows BETTER performance (ΔR² = +{delta_r2:.4f})")
else:
    print(f"   External cohort shows WORSE performance (ΔR² = {delta_r2:.4f})")

print(f"\nBootstrap Details:")
print(f"   Mean ΔR²: {delta_boot.mean():.4f}")
print(f"   95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"   P-value: {p_value:.4f}")

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Set graph parameters
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# Calculate performance metrics
r2_ext = r2_score(y_ext, y_pred_ext)
rmse_ext = np.sqrt(mean_squared_error(y_ext, y_pred_ext))
mae_ext = mean_absolute_error(y_ext, y_pred_ext)

# Calculate MAPE
def calculate_mape(y_true, y_pred):
    eps = 1e-9
    y_true_safe = np.clip(np.asarray(y_true), eps, None)
    y_pred_safe = np.clip(y_pred, eps, None)
    return np.mean(np.abs((y_true_safe - y_pred_safe) / y_true_safe)) * 100

mape_ext = calculate_mape(y_ext, y_pred_ext)

print(f"📊 External validation performance metrics:")
print(f"R²:   {r2_ext:.4f}")
print(f"RMSE: {rmse_ext:.4f}")
print(f"MAE:  {mae_ext:.4f}")
print(f"MAPE: {mape_ext:.2f}%")

# Use only external validation data
scale_factor = 1.5
confidence = 0.95

# Fit the regression line for the external validation set
z_ext = np.polyfit(y_ext, y_pred_ext, 1)
p_ext = np.poly1d(z_ext)
x_extended_ext = np.linspace(min(y_ext), max(y_ext), 100)
predicted_extended_ext = p_ext(x_extended_ext)
residuals_ext = y_pred_ext - p_ext(y_ext)
mean_error_ext = np.mean(residuals_ext**2)
t_value_ext = stats.t.ppf((1 + confidence) / 2., len(y_ext) - 1)
ci_extended_ext = t_value_ext * scale_factor * np.sqrt(mean_error_ext) * np.sqrt(1 / len(y_ext) + (x_extended_ext - np.mean(y_ext))**2 / np.sum((y_ext - np.mean(y_ext))**2))

# Purple color scheme
external_color = '#6A0DAD'  # Dark purple - primary color
confidence_color = '#CBC3E3'  # Light purple - confidence interval
regression_line_color = '#8A2BE2'  # Blue-purple - regression line

fig, ax = plt.subplots(figsize=(8, 6), dpi=1200)

# Plot external validation scatter
scatter = ax.scatter(y_ext, y_pred_ext, color=external_color, alpha=0.7, s=40,
                     label='External Validation Points', edgecolors='white', linewidth=0.5)

# Plot regression line
ax.plot(x_extended_ext, predicted_extended_ext, color=regression_line_color, linewidth=2.5,
        label=f'Regression Line (R² = {r2_ext:.3f})')

# Plot confidence interval
ax.fill_between(x_extended_ext, predicted_extended_ext - ci_extended_ext,
                predicted_extended_ext + ci_extended_ext, color=confidence_color,
                alpha=0.5, label='95% Confidence Band')

# 1:1 reference line
min_val = min(y_ext.min(), y_pred_ext.min())
max_val = max(y_ext.max(), y_pred_ext.max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=1.5, label='1:1 Reference Line')

# Graph settings
ax.set_xlabel('Observed Values', fontsize=14, weight='bold')
ax.set_ylabel('Predicted Values', fontsize=14, weight='bold')
ax.set_title('External Validation: Model Performance on Independent Cohort',
             fontsize=16, weight='bold', pad=20)
ax.legend(loc='upper left', fontsize=11, frameon=True, fancybox=True, shadow=True)
ax.grid(True, linestyle='--', alpha=0.3)

# Add performance metrics text box - with purple border
metrics_text = (
    f"External Validation Cohort\n"
    f"N = {len(y_ext)}\n"
    f"R² = {r2_ext:.3f}\n"
    f"RMSE = {rmse_ext:.3f}\n"
    f"MAE = {mae_ext:.3f}\n"
    f"MAPE = {mape_ext:.1f}%"
)
ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=external_color,
                 alpha=0.9, linewidth=2))

plt.tight_layout()
plt.savefig('external_validation_purple_theme.pdf', format='pdf', bbox_inches='tight', dpi=1200)
plt.show()

print("✅ The external validation chart with purple theme has been generated!")
