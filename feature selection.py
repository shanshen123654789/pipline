import os
os.chdir("/home/jiangshan/cre/growth")
current_path = os.getcwd()
print("current_path :", current_path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
df = pd.read_excel('growth.xlsx')
df

from sklearn.model_selection import train_test_split
X = df.drop(['Y'], axis=1)
y = df['Y']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

print(f'train_data size: {X_train.shape}, test size:: {X_test.shape}')

### 1. Feature selection
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
import os

X = df.drop(['Y'], axis=1)
y = df['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def calculate_median_iqr(data, features):
    stats = {}
    for feature in features:
        median = data[feature].median()
        q1 = data[feature].quantile(0.25)
        q3 = data[feature].quantile(0.75)
        stats[feature] = f"{median} ({q1}-{q3})"
    return stats


continuous_features = ["30kg ABW", "Birth weight", "Litter size"]

train_stats = calculate_median_iqr(X_train, continuous_features)
test_stats = calculate_median_iqr(X_test, continuous_features)
overall_stats = calculate_median_iqr(X, continuous_features)

def calculate_percentage(data, feature):
    return data[feature].value_counts(normalize=True) * 100

categorical_features = ["Sex", "Season", "Parity", "LTC", "RTC"]
train_cat_stats = {feature: calculate_percentage(X_train, feature) for feature in categorical_features}
test_cat_stats = {feature: calculate_percentage(X_test, feature) for feature in categorical_features}
overall_cat_stats = {feature: calculate_percentage(X, feature) for feature in categorical_features}

def calculate_p_value(train_data, test_data, feature):
    _, p_value = stats.ks_2samp(train_data[feature], test_data[feature])
    return p_value

p_values = {feature: calculate_p_value(X_train, X_test, feature) for feature in
            continuous_features + categorical_features}

summary = []
for feature in continuous_features:
    summary.append({
        'Characteristic': feature,
        'Overall': overall_stats.get(feature, ""),
        'Training-set': train_stats.get(feature, ""),
        'Test-set': test_stats.get(feature, ""),
        'P': p_values.get(feature, "")
    })

for feature in categorical_features:
    train_percent = train_cat_stats.get(feature, {})
    test_percent = test_cat_stats.get(feature, {})
    overall_percent = overall_cat_stats.get(feature, {})
    p_value = p_values.get(feature, "")

    overall_percent_str = ", ".join([f"{k}: {v:.1f}%" for k, v in overall_percent.items()])
    train_percent_str = ", ".join([f"{k}: {v:.1f}%" for k, v in train_percent.items()])
    test_percent_str = ", ".join([f"{k}: {v:.1f}%" for k, v in test_percent.items()])

    summary.append({
        'Characteristic': feature,
        'Overall': overall_percent_str,
        'Training-set': train_percent_str,
        'Test-set': test_percent_str,
        'P': p_value
    })

summary_df = pd.DataFrame(summary)

output_dir = "/home/jiangshan/cre/growth" 
os.makedirs(output_dir, exist_ok=True)

file_path = os.path.join(output_dir, "feature_summary.xlsx")
summary_df.to_excel(file_path, index=False)

file_path

import lightgbm as lgb

lgbm_reg = lgb.LGBMRegressor(random_state=42, verbose=-1)
lgbm_reg.fit(X_train, y_train)
feature_importances = lgbm_reg.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)


top_n =8
top_features = feature_importance_df.head(top_n)

plt.figure(figsize=(12, 8))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title( 'Feature Importance', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gca().invert_yaxis()
plt.savefig("lightgbm-Feature_Importance.pdf", format='pdf', dpi=1200, bbox_inches='tight')
plt.savefig("lightgbm-Feature_Importance.jpg", format='jpg', dpi=1200, bbox_inches='tight', pil_kwargs={'quality': 100})
plt.show()

from sklearn.metrics import r2_score


selection_results = pd.DataFrame(columns=['Feature', 'Importance', 'R2'])
selected_features = []

for i in range(len(top_features)):
    current_feature = top_features.iloc[i]['Feature']
    selected_features.append(current_feature)
    X_train_subset = X_train[selected_features]
    X_test_subset = X_test[selected_features]

    lgbm_reg = lgb.LGBMRegressor(random_state=42, verbose=-1)
    lgbm_reg.fit(X_train_subset, y_train)

    y_pred = lgbm_reg.predict(X_test_subset)
    r2_score_value = r2_score(y_test, y_pred)

    selection_results.loc[len(selection_results)] = [
        current_feature,
        top_features.iloc[i]['Importance'],
        r2_score_value
    ]
selection_results

import pandas as pd
from pathlib import Path

out_path = Path("/home/jiangshan/cre/growth/selection_results.xlsx")
out_path.parent.mkdir(parents=True, exist_ok=True)
selection_results.to_excel(out_path, sheet_name="Results", index=False)

print(f"Saved to: {out_path}")

selection_results['Importance'] = (
    selection_results['Importance'] / selection_results['Importance'].sum()
)
selection_results

n_features = 6
fig, ax1 = plt.subplots(figsize=(16, 6))

norm = plt.Normalize(selection_results['Importance'].min(), selection_results['Importance'].max())
colors = plt.cm.Blues(norm(selection_results['Importance']))

ax1.bar(selection_results['Feature'], selection_results['Importance'], color=colors, label='Feature Importance')
ax1.set_xlabel("Features", fontsize=18, fontweight='bold')
ax1.set_ylabel("Feature Importance", fontsize=18, fontweight='bold')
ax1.tick_params(axis='y', labelsize=12, width=1.5)


x_labels = selection_results['Feature']
x_colors = ['red' if i < n_features else 'black' for i in range(len(x_labels))]
for tick_label, color in zip(ax1.get_xticklabels(), x_colors):
    tick_label.set_color(color)

ax1.tick_params(axis='x', rotation=45, labelsize=16, width=1.5)
ax2 = ax1.twinx()

ax2.plot(
    selection_results['Feature'][:n_features + 1],
    selection_results['R2'][:n_features + 1],
    color="red", marker='o', linestyle='-', label="Cumulative R² (Top Features)"
)

# Black dots and black lines: Other features
ax2.plot(
    selection_results['Feature'][n_features:],
    selection_results['R2'][n_features:],
    color="black", marker='o', linestyle='-', label="Cumulative R² (Other Features)"
)

ax2.set_ylabel("Cumulative R²", fontsize=18, fontweight='bold')
ax2.tick_params(axis='y', labelsize=12, width=1.5)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))  # Keep 3 decimal places


plt.title(f"Feature Contribution and R² Performance (Top {n_features} Features Highlighted)", fontsize=18, fontweight='bold')
fig.tight_layout()

plt.savefig("lightgbm-Feature.pdf", format='pdf', dpi=1200, bbox_inches='tight')
plt.savefig("lightgbm-Feature.jpg", format='jpg', dpi=1200, bbox_inches='tight', pil_kwargs={'quality': 100})
plt.show()

from sklearn.model_selection import KFold
selection_results = pd.DataFrame(columns=['Feature', 'Importance', 'Mean_R2'])


selected_features = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)
n_splits = kf.get_n_splits()  # Get the number of folds

fold_columns = [f'Fold_{i+1}_R2' for i in range(n_splits)]

for i in range(len(top_features)):
    # Current feature
    current_feature = top_features.iloc[i]['Feature']
    selected_features.append(current_feature)

    fold_r2_scores = []

    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx][selected_features], X_train.iloc[val_idx][selected_features]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        lgbm_reg = lgb.LGBMRegressor(random_state=42, verbose=-1)
        lgbm_reg.fit(X_train_fold, y_train_fold)

        y_val_pred = lgbm_reg.predict(X_val_fold)
        fold_r2_score = r2_score(y_val_fold, y_val_pred)
        fold_r2_scores.append(fold_r2_score)

    mean_r2_score = np.mean(fold_r2_scores)

    row_data = {
        'Feature': current_feature,
        'Importance': top_features.iloc[i]['Importance'],
        'Mean_R2': mean_r2_score,
    }

    for j, score in enumerate(fold_r2_scores):
        row_data[fold_columns[j]] = score

    row_df = pd.DataFrame([row_data])
    selection_results = pd.concat([selection_results, row_df], ignore_index=True)

selection_results.to_csv('selection_results.csv', index=False, encoding='utf-8-sig')

selection_results

selection_results['Importance'] = (
    selection_results['Importance'] / selection_results['Importance'].sum()
)
selection_results


import scipy.stats as stats

fold_columns = [col for col in selection_results.columns if 'Fold_' in col]

selection_results['CI_Lower'] = None
selection_results['CI_Upper'] = None

for index, row in selection_results.iterrows():
    fold_r2_scores = [row[fold] for fold in fold_columns]
    n_folds = len(fold_r2_scores)
    mean_r2 = row['Mean_R2']
    std_err = stats.sem(fold_r2_scores)
    t_value = stats.t.ppf(0.975, df=n_folds - 1)
    ci_lower = mean_r2 - t_value * std_err
    ci_upper = mean_r2 + t_value * std_err

    selection_results.at[index, 'CI_Lower'] = ci_lower
    selection_results.at[index, 'CI_Upper'] = ci_upper

selection_results
selection_results.to_csv('selection_results_with_ci.csv', index=False, encoding='utf-8-sig')
selection_results

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

mpl.rcParams['font.family'] = 'Times New Roman'  # Global font
mpl.rcParams['pdf.fonttype'] = 42  # Use TrueType font for easy editing in AI/PS
mpl.rcParams['ps.fonttype'] = 42

selection_results['Importance'] = pd.to_numeric(selection_results['Importance'], errors='coerce')
selection_results['CI_Lower'] = pd.to_numeric(selection_results['CI_Lower'], errors='coerce')
selection_results['CI_Upper'] = pd.to_numeric(selection_results['CI_Upper'], errors='coerce')

n_features = 6
fig, ax1 = plt.subplots(figsize=(16, 6))

norm = plt.Normalize(selection_results['Importance'].min(), selection_results['Importance'].max())
colors = plt.cm.Blues(norm(selection_results['Importance']))

ax1.bar(selection_results['Feature'], selection_results['Importance'], color=colors, label='Feature Importance')
ax1.set_xlabel("Features", fontsize=18, fontweight='bold')
ax1.set_ylabel("Feature Importance", fontsize=18, fontweight='bold')
ax1.tick_params(axis='y', labelsize=12, width=1.5)

x_labels = selection_results['Feature']
x_colors = ['red' if i < n_features else 'black' for i in range(len(x_labels))]
ax1.set_xticks(range(len(x_labels)))
ax1.set_xticklabels(x_labels)

for tick_label, color in zip(ax1.get_xticklabels(), x_colors):
    tick_label.set_color(color)

ax1.tick_params(axis='x', rotation=45, labelsize=16, width=1.5)
ax2 = ax1.twinx()


ax2.plot(
    selection_results['Feature'][:n_features + 1],  # Transition from red to black points
    selection_results['Mean_R2'][:n_features + 1],
    color="red", marker='o', linestyle='-', label="Mean R² (Top Features)"
)

ax2.plot(
    selection_results['Feature'][n_features:],
    selection_results['Mean_R2'][n_features:],
    color="black", marker='o', linestyle='-', label="Mean R² (Other Features)"
)

ax2.fill_between(
    selection_results['Feature'],
    selection_results['CI_Lower'],
    selection_results['CI_Upper'],
    color='red',
    alpha=0.2,
)

ax2.set_ylabel("Mean R²", fontsize=18, fontweight='bold')
ax2.tick_params(axis='y', labelsize=12, width=1.5)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))

plt.title(f"Feature Contribution and R² Performance (Top {n_features} Features Highlighted)",
          fontsize=18, fontweight='bold')

fig.tight_layout()

pdf_path = "feature_importance_R2_with_CI.pdf"
jpg_path = "feature_importance_R2_with_CI.jpg"


fig.savefig(pdf_path, format="pdf", dpi=1200, bbox_inches="tight")
fig.savefig(jpg_path, format="jpg", dpi=600, bbox_inches="tight")
plt.show()
plt.close(fig)
