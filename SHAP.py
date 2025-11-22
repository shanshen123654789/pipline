import sys
import os
# Redirect standard output to os.devnull to avoid excessive SHAP output
sys.stdout = open(os.devnull, 'w')
import shap

# Use TreeExplainer to explain the CatBoost model
explainer = shap.TreeExplainer(catboost)

# Calculate SHAP values as a numpy.array
shap_values_numpy = explainer.shap_values(X_test)

# Restore standard output
sys.stdout.close()
sys.stdout = sys.__stdout__

print("✅ SHAP values calculation completed!")
print(f"SHAP values array shape: {shap_values_numpy.shape}")

import matplotlib.pyplot as plt
import numpy as np

# Set global font and style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

print("Starting to create SHAP visualizations...")

# ===== 1. SHAP Bee Swarm Plot (Dot Plot) =====
print("📈 Creating SHAP Bee Swarm Plot...")
fig1, ax1 = plt.subplots(figsize=(10, 8))

# Plot the Bee Swarm Plot
shap.summary_plot(shap_values_numpy, X_test,
                 feature_names=X_test.columns,
                 plot_type="dot",
                 show=False,
                 color_bar=True)

# Beautify the plot
ax1.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Features', fontsize=12, fontweight='bold')
ax1.set_title('SHAP Bee Swarm Plot\n(Feature Impact Distribution)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("/home/jiangshan/cre/growth2/SHAP_BeeSwarm_Plot.pdf",
            format='pdf', bbox_inches='tight', dpi=1200)
plt.show()

print("✅ SHAP Bee Swarm Plot has been saved")

# ===== 2. SHAP Feature Importance Bar Chart =====
print("📊 Creating SHAP Feature Importance Bar Chart...")
fig2, ax2 = plt.subplots(figsize=(10, 8))

# Calculate the average SHAP value for each feature
feature_importance = np.abs(shap_values_numpy).mean(0)
feature_names = X_test.columns

# Combine feature names and their corresponding average SHAP values into a DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

# Sort by average SHAP values in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=True)  # Sort in ascending order because barh reverses the order

# Create a color mapping - Ensure that the most important features are purple
# Use the color sequence you provided but make sure purple is assigned to the most important features
colors = ['#800080', '#9370DB', '#BA55D3', '#DDA0DD', '#E6E6FA', '#FFB6C1', '#FFC0CB', '#FFDAB9', '#FFF0F5']

# Adjust color list based on the number of features
if len(importance_df) <= len(colors):
    used_colors = colors[:len(importance_df)]
else:
    # If the number of features exceeds the number of colors, repeat the colors
    used_colors = colors * (len(importance_df) // len(colors) + 1)
    used_colors = used_colors[:len(importance_df)]

# Reverse the color order so the most important features are purple
used_colors = list(reversed(used_colors))

# Plot the bar chart - directly assign colors
bars = ax2.barh(importance_df['Feature'], importance_df['Importance'],
                color=used_colors)

# Beautify the chart
ax2.set_xlabel('Mean |SHAP Value| (Average Impact Magnitude)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Features', fontsize=12, fontweight='bold')
ax2.set_title('SHAP Feature Importance\n(Mean Absolute Impact)', fontsize=14, fontweight='bold')

# Add grid lines for better readability
ax2.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig("/home/jiangshan/cre/growth2/SHAP_Feature_Importance.pdf",
            format='pdf', bbox_inches='tight', dpi=1200)
plt.savefig("/home/jiangshan/cre/growth2/SHAP_Feature_Importance.jpg",
            format='jpg',
            dpi=1200,
            bbox_inches='tight',
            pil_kwargs={'quality': 100})
plt.show()

print("✅ SHAP Feature Importance chart saved")


### Calculate SHAP values in Explanation format
shap_values_Explanation = explainer(X_test)
feature_name = '30kg ABW'
# Find the index of the specified feature
feature_index = shap_values_Explanation.feature_names.index(feature_name)

plt.figure(figsize=(10, 5))
shap.plots.scatter(shap_values_Explanation[:, feature_index], show=False)
plt.title(f'SHAP Scatter Plot for Feature: {feature_name}')
plt.savefig("30kg_ABW.pdf", format='pdf', bbox_inches='tight', dpi=1200)
plt.tight_layout()
plt.show()


shap_values_df = pd.DataFrame(shap_values_numpy, columns=X.columns)
shap_values_df.head()

from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d

# Selected feature
feature_to_plot = '30kg ABW'

# Extract data
X_data = X_test[feature_to_plot].values
Y_data = shap_values_df[feature_to_plot].values

# LOWESS fitting
lowess_frac = 0.4
sort_idx = np.argsort(X_data)
X_sorted = X_data[sort_idx]
Y_sorted = Y_data[sort_idx]

# LOWESS fitting
lowess_fit_points = lowess(Y_sorted, X_sorted, frac=lowess_frac, it=0)

# Generate unique X values and corresponding Y values
unique_x, unique_indices = np.unique(lowess_fit_points[:, 0], return_index=True)
y_for_unique_x = lowess_fit_points[unique_indices, 1]
lowess_interpolation_function = interp1d(unique_x, y_for_unique_x, kind='linear', fill_value="extrapolate")

# Calculate R² value
y_pred = lowess_interpolation_function(X_sorted)
y_true = Y_sorted
ss_res = np.sum((y_true - y_pred)**2)
ss_tot = np.sum((y_true - np.mean(y_true))**2)
r_squared = 1.0 - (ss_res / ss_tot)

# LOWESS confidence interval (Bootstrapping)
n_bootstraps = 100  # Set the number of Bootstrap samples
random_seed = 42  # Set the random seed for reproducibility
np.random.seed(random_seed)  # Fix the random seed

# Set the confidence interval range
x_min_plot_ci, x_max_plot_ci = X_data.min(), X_data.max()
x_smooth_for_ci = np.linspace(x_min_plot_ci, x_max_plot_ci, 100)

boot_lowess_preds = []  # Store LOWESS predictions for each bootstrap sample

# Perform Bootstrap sampling
for i in range(n_bootstraps):
    # Perform resampling with replacement
    indices = np.random.choice(len(X_data), len(X_data), replace=True)
    X_boot, Y_boot = X_data[indices], Y_data[indices]

    # Sort data
    sort_idx_boot = np.argsort(X_boot)
    X_boot_sorted, Y_boot_sorted = X_boot[sort_idx_boot], Y_boot[sort_idx_boot]

    if len(X_boot_sorted) < 2:  # Skip if there are fewer than 2 data points
        continue

    # Perform LOWESS fitting for the bootstrap sample
    current_boot_lowess = lowess(Y_boot_sorted, X_boot_sorted, frac=lowess_frac, it=0)

    # Get unique X values and corresponding Y values
    unique_x_boot, unique_idx_boot = np.unique(current_boot_lowess[:, 0], return_index=True)
    y_for_unique_x_boot = current_boot_lowess[unique_idx_boot, 1]

    # Linear interpolation
    y_interp_boot = np.interp(x_smooth_for_ci, unique_x_boot, y_for_unique_x_boot)
    boot_lowess_preds.append(y_interp_boot)

# Calculate confidence intervals
if boot_lowess_preds:
    boot_lowess_preds_array = np.array(boot_lowess_preds)
    ci_lower = np.percentile(boot_lowess_preds_array, 2.5, axis=0)  # Calculate 2.5th percentile
    ci_upper = np.percentile(boot_lowess_preds_array, 97.5, axis=0)  # Calculate 97.5th percentile
    plot_ci = True  # Enable confidence interval plotting
else:
    plot_ci = False

# Plot
fig, ax = plt.subplots(figsize=(10, 5.5))

# Scatter plot
scatter_plot = ax.scatter(X_data, Y_data, s=40, alpha=0.7, edgecolor='none', zorder=2, label='Sample')

# Plot LOWESS trend line
ax.plot(lowess_fit_points[:, 0], lowess_fit_points[:, 1], color='#A52A2A', linewidth=2.5, label=f'Fit line ($R^2$ = {r_squared:.3f})', zorder=3)

# Plot LOWESS 95% confidence interval
if plot_ci:
    ax.fill_between(x_smooth_for_ci, ci_lower, ci_upper, color='salmon', alpha=0.35, label='95% CI', zorder=1)

# Set labels and ticks
ax.set_xlabel(feature_to_plot.capitalize(), fontsize=18, fontweight='bold')
ax.set_ylabel("SHAP value", fontsize=18, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=18, width=2)

# Axis lines
ax.spines['top'].set_linewidth(1.2)
ax.spines['right'].set_linewidth(1.2)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# Legend
handles, labels = ax.get_legend_handles_labels()
if handles:
    ax.legend(loc='upper right', fontsize=18, prop={'weight': 'bold'}, frameon=True, facecolor='white', framealpha=0.7, edgecolor='lightgray')

# Axis limits
ax.set_xlim(left=X_data.min() - (X_data.max() - X_data.min()) * 0.05, right=X_data.max() * 1.02)
y_min, y_max = np.nanmin(Y_data), np.nanmax(Y_data)
ax.set_ylim(bottom=y_min - (y_max - y_min) * 0.15, top=y_max + (y_max - y_min) * 0.15)

# Add horizontal line at SHAP = 0
plt.axhline(y=0, color='black', linestyle='-.', linewidth=1, label='SHAP = 0')

# Bin the X_data range, calculate scatter counts per bin
n_bins = 20  # Set the number of bins
bin_edges = np.linspace(X_data.min(), X_data.max(), n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate the center value of each bin
counts, _ = np.histogram(X_data, bins=bin_edges)  # Calculate the count of points in each bin

# Add bar chart
ax2 = ax.twinx()  # Create a second y-axis
ax2.bar(bin_centers, counts, width=np.diff(bin_edges), color='lightblue', alpha=0.2, align='center', zorder=1)  # Plot bar chart
ax2.set_ylabel('Distribution', fontsize=18, fontweight='bold')
ax2.tick_params(axis='both', which='major', labelsize=18)
ax2.spines['right'].set_linewidth(1.2)
ax2.spines['right'].set_color('gray')

# Add dashed line for the fit line
x_fit_line_at_0 = lowess_fit_points[:, 0][np.abs(lowess_fit_points[:, 1]).argmin()]  # Find x value where fit line is closest to y=0
ax.axvline(x=x_fit_line_at_0, color='gray', linestyle='--', linewidth=1)

# Annotate the x value next to the dashed line
ax.text(x_fit_line_at_0, y_min - (y_max - y_min) * 0.1, f'x = {x_fit_line_at_0:.2f}', color='gray', fontsize=18, fontweight='bold', horizontalalignment='center')

# Add dashed line for y=0
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)

# Adjust layout to fit color bar and y-axis bar chart
plt.tight_layout(rect=[0, 0.03, 0.9, 0.97])  # Adjust the rect's third parameter to make space for the color bar
plt.savefig("30kg_ABW_SHAP.pdf", format='pdf', bbox_inches='tight')
plt.show()

from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d
import shap
import numpy as np
import matplotlib.pyplot as plt

# Assume you have already trained a model and calculated SHAP values
# Calculate SHAP values
explainer = shap.Explainer(catboost)  # Using your trained model
shap_values = explainer(X_test)  # Compute SHAP values

# Convert SHAP values to DataFrame
shap_values_df = shap_values.values  # Get SHAP value array
shap_values_df = pd.DataFrame(shap_values_df, columns=X_test.columns)  # Convert to DataFrame

# List of features to analyze
features_to_plot = ['30kg ABW', 'Litter size', 'Season', 'Birth weight', 'Parity', 'Sex']

# Create figure and subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Create a 2-row, 3-column subplot

# Iterate through each feature
for i, feature_to_plot in enumerate(features_to_plot):
    # Get the current subplot
    ax = axes[i//3, i%3]  # Calculate the current subplot position (i//3 is row, i%3 is column)

    # Extract data
    X_data = X_test[feature_to_plot].values
    Y_data = shap_values_df[feature_to_plot].values

    # LOWESS fitting
    lowess_frac = 0.4
    sort_idx = np.argsort(X_data)
    X_sorted = X_data[sort_idx]
    Y_sorted = Y_data[sort_idx]

    # LOWESS fitting
    lowess_fit_points = lowess(Y_sorted, X_sorted, frac=lowess_frac, it=0)

    # Generate unique X values and corresponding Y values
    unique_x, unique_indices = np.unique(lowess_fit_points[:, 0], return_index=True)
    y_for_unique_x = lowess_fit_points[unique_indices, 1]
    lowess_interpolation_function = interp1d(unique_x, y_for_unique_x, kind='linear', fill_value="extrapolate")

    # Calculate R² value
    y_pred = lowess_interpolation_function(X_sorted)
    y_true = Y_sorted
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r_squared = 1.0 - (ss_res / ss_tot)

    # LOWESS confidence interval (Bootstrapping)
    n_bootstraps = 100  # Set the number of Bootstrap samples
    random_seed = 42  # Set the random seed for reproducibility
    np.random.seed(random_seed)  # Fix the random seed

    # Set the confidence interval range
    x_min_plot_ci, x_max_plot_ci = X_data.min(), X_data.max()
    x_smooth_for_ci = np.linspace(x_min_plot_ci, x_max_plot_ci, 100)

    boot_lowess_preds = []  # Store LOWESS predictions for each bootstrap sample

    # Perform Bootstrap sampling
    for _ in range(n_bootstraps):
        indices = np.random.choice(len(X_data), len(X_data), replace=True)
        X_boot, Y_boot = X_data[indices], Y_data[indices]

        # Sort data
        sort_idx_boot = np.argsort(X_boot)
        X_boot_sorted, Y_boot_sorted = X_boot[sort_idx_boot], Y_boot[sort_idx_boot]

        if len(X_boot_sorted) < 2:
            continue

        # Perform LOWESS fitting for the bootstrap sample
        current_boot_lowess = lowess(Y_boot_sorted, X_boot_sorted, frac=lowess_frac, it=0)

        # Get unique X values and corresponding Y values
        unique_x_boot, unique_idx_boot = np.unique(current_boot_lowess[:, 0], return_index=True)
        y_for_unique_x_boot = current_boot_lowess[unique_idx_boot, 1]

        # Linear interpolation
        y_interp_boot = np.interp(x_smooth_for_ci, unique_x_boot, y_for_unique_x_boot)
        boot_lowess_preds.append(y_interp_boot)

    # Calculate confidence interval
    if boot_lowess_preds:
        boot_lowess_preds_array = np.array(boot_lowess_preds)
        ci_lower = np.percentile(boot_lowess_preds_array, 2.5, axis=0)
        ci_upper = np.percentile(boot_lowess_preds_array, 97.5, axis=0)
        plot_ci = True
    else:
        plot_ci = False

    # Plot scatter plot
    ax.scatter(X_data, Y_data, s=40, alpha=0.7, edgecolor='none', zorder=2, label='Sample')

    # Plot LOWESS trend line
    ax.plot(lowess_fit_points[:, 0], lowess_fit_points[:, 1], color='#A52A2A', linewidth=2.5, label=f'Fit line ($R^2$ = {r_squared:.3f})', zorder=3)

    # Plot LOWESS 95% confidence interval
    if plot_ci:
        ax.fill_between(x_smooth_for_ci, ci_lower, ci_upper, color='salmon', alpha=0.35, label='95% CI', zorder=1)

    # Set labels and ticks
    ax.set_xlabel(feature_to_plot.capitalize(), fontsize=18, fontweight='bold')
    ax.set_ylabel("SHAP value", fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=18, width=2)

    # Axis lines
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper right', fontsize=18, prop={'weight': 'bold'}, frameon=True, facecolor='white', framealpha=0.7, edgecolor='lightgray')

    # Axis limits
    ax.set_xlim(left=X_data.min() - (X_data.max() - X_data.min()) * 0.05, right=X_data.max() * 1.02)
    y_min, y_max = np.nanmin(Y_data), np.nanmax(Y_data)
    ax.set_ylim(bottom=y_min - (y_max - y_min) * 0.15, top=y_max + (y_max - y_min) * 0.15)

    # Add horizontal line at SHAP = 0
    ax.axhline(y=0, color='black', linestyle='-.', linewidth=1, label='SHAP = 0')

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# Save the figure as a PDF
plt.savefig("SHAP_Plots_All_Features.pdf", format='pdf', bbox_inches='tight')

# Display the figure
plt.show()

import shap

# Create the explainer - using your trained catboost model
explainer = shap.TreeExplainer(catboost)  # Replace best_model_xgb with catboost

# Calculate SHAP interaction values
shap_interaction_values = explainer.shap_interaction_values(X_test)
shap_values_numpy = explainer.shap_values(X_test)

print("✅ SHAP values computation completed!")
print(f"SHAP interaction values shape: {np.array(shap_interaction_values).shape}")
print(f"SHAP values shape: {np.array(shap_values_numpy).shape}")

# Get the feature names
feature_names = X_test.columns

# Initialize an empty DataFrame, the size of the matrix is the number of features
interaction_matrix = pd.DataFrame(np.nan, index=feature_names, columns=feature_names)

# Iterate over each pair of features
for i, feature in enumerate(feature_names):
    for j, other_feature in enumerate(feature_names):
        if i != j:  # Only consider interactions between different features
            # Calculate the absolute SHAP interaction values for each sample
            interaction_values = [shap_interaction_values[sample_idx][i, j] for sample_idx in range(shap_interaction_values.shape[0])]
            # Calculate the mean absolute value of the interaction values
            avg_interaction_value = np.mean(np.abs(interaction_values))
            # Assign the result to the corresponding matrix position
            interaction_matrix.loc[feature, other_feature] = avg_interaction_value
        else:
            # Calculate the main effect values (diagonal values)
            main_effect_values = [shap_interaction_values[sample_idx][i, i] for sample_idx in range(shap_interaction_values.shape[0])]
            # Calculate the mean absolute value of the main effect values
            avg_main_effect_value = np.mean(np.abs(main_effect_values))
            # Assign the main effect value to the diagonal position
            interaction_matrix.loc[feature, feature] = avg_main_effect_value

# Output the final interaction_matrix
interaction_matrix

# Create a mask: set the diagonal part to NaN
mask = np.eye(len(interaction_matrix), dtype=bool)

# Set the diagonal part to NaN in order to calculate the minimum and maximum values for the non-diagonal parts
interaction_matrix_no_diag = interaction_matrix.copy()
interaction_matrix_no_diag[mask] = np.nan

# Get the minimum and maximum values for the non-diagonal part
vmin = interaction_matrix_no_diag.min().min() * 2  # Minimum value
vmax = interaction_matrix_no_diag.max().max() * 2  # Maximum value

# Create figure and axis - increase figure size
fig, ax = plt.subplots(figsize=(12, 10), dpi=1200)

# Use the 'viridis' colormap
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=vmin, vmax=vmax)  # Set the range based on the non-diagonal minimum and maximum values

# Initialize an empty list to store scatter objects (for color bar)
scatter_handles = []

# Calculate the scaling factor for bubble size - dynamically adjust based on the maximum value in the matrix
max_abs_value = np.nanmax(np.abs(interaction_matrix_no_diag)) * 2
# Dynamically adjust the scaling factor to ensure bubbles are not too large
size_factor = 3000 / max_abs_value if max_abs_value > 0 else 1000

# Loop to plot bubbles and values
for i in range(len(interaction_matrix.columns)):
    for j in range(len(interaction_matrix.columns)):
        # Multiply the interaction matrix value by 2 to calculate the color and bubble size
        value = interaction_matrix.iloc[i, j] * 2

        if i > j:  # Bottom-left part of the diagonal, show only bubbles
            color = cmap(norm(value))  # Get the color based on the interaction coefficient
            # Use the dynamically adjusted scaling factor
            scatter = ax.scatter(i, j, s=np.abs(value) * size_factor, color=color, alpha=0.75)
            scatter_handles.append(scatter)  # Save scatter object for color bar
        elif i < j:  # Top-right part of the diagonal, show only values
            color = cmap(norm(value))  # The color of the values is also based on the interaction coefficient
            ax.text(i, j, f'{value:.3f}', ha='center', va='center', color=color, fontsize=15)
        else:  # Diagonal part, show blank
            ax.scatter(i, j, s=1, color='white')

# Set axis labels
ax.set_xticks(range(len(interaction_matrix.columns)))
ax.set_xticklabels(interaction_matrix.columns, rotation=45, ha='right', fontsize=18)
ax.set_yticks(range(len(interaction_matrix.columns)))
ax.set_yticklabels(interaction_matrix.columns, fontsize=18)

# Set axis limits to ensure enough space for bubbles
ax.set_xlim(-0.5, len(interaction_matrix.columns) - 0.5)
ax.set_ylim(-0.5, len(interaction_matrix.columns) - 0.5)

# Add grid lines for better alignment
ax.grid(False)  # Do not display grid lines, but keep axis ticks

# Add color bar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only used for displaying color bar
cbar = fig.colorbar(sm, ax=ax, label='SHAP interaction value')

# Set color bar tick font and label font
cbar.ax.tick_params(labelsize=18)  # Set color bar tick font size to 18
cbar.set_label('SHAP interaction value', fontsize=18, fontweight='bold')  # Set label font size to 18 and bold

# Adjust layout to avoid clipping the image - increase margin
plt.tight_layout(pad=3.0)
plt.savefig("SHAP_interaction_value.pdf", format='pdf', bbox_inches='tight', dpi=1200)
plt.show()

# Use X_test.columns as feature names
feature_names = X_test.columns

# Initialize a dictionary to store DataFrames
dataframes = {}

# Iterate through each feature
for i, feature in enumerate(feature_names):
    # Initialize a DataFrame to store the interaction values between the current feature and other features
    interaction_dict = {f"{other_feature}": [] for j, other_feature in enumerate(feature_names) if j != i}

    # Iterate through each sample and extract interaction values between the current feature and other features
    for sample_idx in range(shap_interaction_values.shape[0]):
        # Interaction matrix for the current sample
        interaction_matrix = shap_interaction_values[sample_idx]

        # Extract interaction values between the current feature and other features
        interactions = [interaction_matrix[i, j] for j in range(len(feature_names)) if j != i]

        # Append to the corresponding column
        for col_idx, col_name in enumerate(interaction_dict.keys()):
            interaction_dict[col_name].append(interactions[col_idx])

    # Create DataFrame
    df = pd.DataFrame(interaction_dict)

    # Rename columns to df_<current_feature_name>
    df.columns = [f"df_{feature}_{col}" for col in interaction_dict.keys()]

    # Store in the dictionary
    dataframes[f"df_{i + 1}"] = df

# Convert DataFrame to global variables
for i, (name, df) in enumerate(dataframes.items()):
    globals()[f"df_{i + 1}"] = df  # Dynamically create variable names

plt.figure(figsize=(6, 4), dpi=1200)
sc = plt.scatter(X_test["30kg ABW"], df_1['df_30kg ABW_Season']*2,
                 s=10, c=X_test["Season"], cmap='inferno_r')
cbar = plt.colorbar(sc, aspect=30, shrink=1)  # Adjust the color bar's aspect ratio and length
cbar.set_label('Season', fontsize=12)  # Set the label for the color bar
cbar.outline.set_visible(False)
plt.axhline(y=0, color='black', linestyle='-.', linewidth=1)
plt.xlabel('30kg ABW', fontsize=12)
plt.ylabel('SHAP interaction value for\n 30kg ABW and Season', fontsize=12)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig("dependence_plot_30kg ABW_Season.pdf", format='pdf', bbox_inches='tight', dpi=1200)
plt.show()

import shap
import matplotlib.pyplot as plt
import numpy as np

# Create explainer
explainer = shap.TreeExplainer(catboost)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)

print("🔍 SHAP Analysis Statistics:")
print(f"Baseline (expected_value): {explainer.expected_value:.4f}")
print(f"SHAP values shape: {shap_values.shape}")
print(f"Number of features: {len(X_test.columns)}")

# 1. Generate global force plot (first 100 samples)
n_samples = 100
sample_indices = range(min(n_samples, len(X_test)))

shap_values_subset = shap_values[sample_indices]
X_test_subset = X_test.iloc[sample_indices]

print(f"\n📊 Generating global SHAP interaction plot (first {len(sample_indices)} samples)...")

force_plot_global = shap.force_plot(
    base_value=explainer.expected_value,
    shap_values=shap_values_subset,
    features=X_test_subset,
    feature_names=X_test.columns.tolist(),
    matplotlib=False,
    show=False
)

shap.save_html("SHAP_Global_Force_Plot.html", force_plot_global, full_html=True)
print("✅ Global SHAP interaction plot saved as SHAP_Global_Force_Plot.html")

# 5. Generate a force plot for a single sample and save as PDF
print("\n🔎 Generating SHAP explanation plot for a single sample...")
sample_indices_individual = [10, 25, 75]  # Select a few samples for detailed analysis

for idx in sample_indices_individual:
    if idx < len(X_test):
        print(f"  Generating SHAP explanation for sample {idx}...")

        # Generate force plot using matplotlib mode
        plt.figure(figsize=(12, 4))
        shap.force_plot(
            base_value=explainer.expected_value,
            shap_values=shap_values[idx],
            features=X_test.iloc[idx],
            feature_names=X_test.columns.tolist(),
            matplotlib=True,  # Use matplotlib mode
            show=False
        )

        # Save as PDF
        plt.tight_layout()
        plt.savefig(f"SHAP_Individual_Sample_{idx}.pdf",
                    format='pdf',
                    bbox_inches='tight',
                    dpi=1200,
                    facecolor='white',
                    edgecolor='none')
        plt.close()  # Close the figure to free memory

        print(f"  ✅ SHAP explanation for sample {idx} has been saved as PDF")

import shap
import matplotlib.pyplot as plt

# Create explainer
explainer = shap.TreeExplainer(catboost)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)

# Choose the 10th sample (index 9)
sample_idx = 9

# Get predicted value and actual value
predicted_value = catboost.predict(X_test.iloc[sample_idx:sample_idx+1])[0]
actual_value = y_test.iloc[sample_idx] if hasattr(y_test, 'iloc') else y_test[sample_idx]
error = abs(actual_value - predicted_value)

print(f"Generating SHAP waterfall plot for sample {sample_idx+1}...")
print(f"Actual value: {actual_value:.4f}, Predicted value: {predicted_value:.4f}, Error: {error:.4f}")

# Create the figure
plt.figure(figsize=(10, 5), dpi=1200)

# Create Explanation object
exp = shap.Explanation(
    values=shap_values[sample_idx],
    base_values=explainer.expected_value,
    data=X_test.iloc[sample_idx],
    feature_names=X_test.columns.tolist()
)

# Plot SHAP waterfall plot
shap.plots.waterfall(exp, show=False, max_display=8)

# Save as PDF and JPG
plt.savefig(f"SHAP_Waterfall_Plot_Sample_{sample_idx+1}.pdf",
           format='pdf',
           bbox_inches='tight')
plt.savefig(f"SHAP_Waterfall_Plot_Sample_{sample_idx+1}.jpg",
           format='jpg',
           dpi=1200,
           bbox_inches='tight',
           pil_kwargs={'quality': 100})

plt.tight_layout()
plt.show()

print(f"✅ SHAP waterfall plot for sample {sample_idx+1} has been saved as PDF and JPG")

import shap
import matplotlib.pyplot as plt

# Create explainer
explainer = shap.TreeExplainer(catboost)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)

# Create shap.Explanation object - For regression models, only one dimension is needed
shap_explanation = shap.Explanation(shap_values,  # For regression models, directly use shap_values
                                    base_values=explainer.expected_value,
                                    data=X_test,
                                    feature_names=X_test.columns)

# Plot heatmap
plt.figure(figsize=(12, 8))
shap.plots.heatmap(shap_explanation, show=False)
plt.tight_layout()
plt.savefig("SHAP_Heatmap_Regression.pdf", format='pdf', bbox_inches='tight', dpi=1200)
plt.savefig("SHAP_Heatmap_Regression.jpg",
            format='jpg',
            dpi=1200,
            bbox_inches='tight',
            pil_kwargs={'quality': 100})
plt.show()

print("✅ SHAP heatmap for regression model has been saved as PDF and JPG")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set figure parameters
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# First, calculate SHAP interaction values
print("🔄 Calculating SHAP interaction values...")

# Use TreeExplainer to explain the CatBoost model
explainer = shap.TreeExplainer(catboost)

# Calculate SHAP interaction values
shap_interaction_values = explainer.shap_interaction_values(X_test)

print(f"✅ SHAP interaction values calculated!")
print(f"Shape of SHAP interaction values array: {shap_interaction_values.shape}")

# Get feature names
feature_names_test = X_test.columns  # Feature names for the test set

# Initialize an empty DataFrame, size of feature number matrix
interaction_matrix_test = pd.DataFrame(np.nan, index=feature_names_test, columns=feature_names_test)

# Loop through each feature pair to calculate interaction effects
print("📊 Constructing interaction matrix...")
for i, feature in enumerate(feature_names_test):
    for j, other_feature in enumerate(feature_names_test):
        if i != j:  # Only consider interactions between different features
            # Calculate the absolute SHAP interaction values for each sample
            interaction_values = [shap_interaction_values[sample_idx][i, j] for sample_idx in range(shap_interaction_values.shape[0])]
            # Calculate the average absolute interaction value
            avg_interaction_value = np.mean(np.abs(interaction_values))
            # Assign the result to the corresponding position in the matrix
            interaction_matrix_test.loc[feature, other_feature] = avg_interaction_value
        else:
            # Calculate main effect values (on the diagonal)
            main_effect_values = [shap_interaction_values[sample_idx][i, i] for sample_idx in range(shap_interaction_values.shape[0])]
            # Calculate the average absolute main effect value
            avg_main_effect_value = np.mean(np.abs(main_effect_values))
            # Assign the main effect value to the diagonal
            interaction_matrix_test.loc[feature, feature] = avg_main_effect_value

# Output the final interaction matrix
print("✅ Interaction matrix construction complete!")
print("\n📋 Interaction matrix:")
print(interaction_matrix_test)

# Visualize the interaction matrix
plt.figure(figsize=(10, 8))
mask = np.zeros_like(interaction_matrix_test, dtype=bool)
np.fill_diagonal(mask, True)  # Mask the diagonal so it displays differently in the heatmap

# Create heatmap
sns.heatmap(interaction_matrix_test,
            annot=True,
            fmt=".4f",
            cmap="YlOrRd",
            cbar_kws={'label': 'Average Absolute SHAP Value'},
            mask=mask,  # Mask diagonal
            square=True)

plt.title("SHAP Feature Interaction Matrix\n(Diagonal: Main Effects, Off-diagonal: Interaction Effects)", fontsize=14, weight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("shap_interaction_matrix.pdf", format='pdf', bbox_inches='tight', dpi=1200)
plt.show()

# Save interaction matrix to CSV file
interaction_matrix_test.to_csv("shap_interaction_matrix.csv")
print(f"💾 Interaction matrix has been saved to 'shap_interaction_matrix.csv'")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set figure parameters
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# First, calculate SHAP interaction values
print("🔄 Calculating SHAP interaction values...")

# Use TreeExplainer to explain the CatBoost model
explainer = shap.TreeExplainer(catboost)

# Calculate SHAP interaction values
shap_interaction_values = explainer.shap_interaction_values(X_test)

print(f"✅ SHAP interaction values calculated!")
print(f"Shape of SHAP interaction values array: {shap_interaction_values.shape}")

# Get feature names
feature_names_test = X_test.columns  # Feature names for the test set

# Initialize an empty DataFrame, size of feature number matrix
interaction_matrix_test = pd.DataFrame(np.nan, index=feature_names_test, columns=feature_names_test)

# Loop through each feature pair to calculate interaction effects
print("📊 Constructing interaction matrix...")
for i, feature in enumerate(feature_names_test):
    for j, other_feature in enumerate(feature_names_test):
        if i != j:  # Only consider interactions between different features
            # Calculate the absolute SHAP interaction values for each sample
            interaction_values = [shap_interaction_values[sample_idx][i, j] for sample_idx in range(shap_interaction_values.shape[0])]
            # Calculate the average absolute interaction value
            avg_interaction_value = np.mean(np.abs(interaction_values))
            # Assign the result to the corresponding position in the matrix
            interaction_matrix_test.loc[feature, other_feature] = avg_interaction_value
        else:
            # Calculate main effect values (on the diagonal)
            main_effect_values = [shap_interaction_values[sample_idx][i, i] for sample_idx in range(shap_interaction_values.shape[0])]
            # Calculate the average absolute main effect value
            avg_main_effect_value = np.mean(np.abs(main_effect_values))
            # Assign the main effect value to the diagonal
            interaction_matrix_test.loc[feature, feature] = avg_main_effect_value

# Output the final interaction matrix
print("✅ Interaction matrix construction complete!")
print("\n📋 Interaction matrix:")
print(interaction_matrix_test)

# Visualize the interaction matrix
plt.figure(figsize=(10, 8))
mask = np.zeros_like(interaction_matrix_test, dtype=bool)
np.fill_diagonal(mask, True)  # Mask the diagonal so it displays differently in the heatmap

# Create heatmap
sns.heatmap(interaction_matrix_test,
            annot=True,
            fmt=".4f",
            cmap="YlOrRd",
            cbar_kws={'label': 'Average Absolute SHAP Value'},
            mask=mask,  # Mask diagonal
            square=True)

plt.title("SHAP Feature Interaction Matrix\n(Diagonal: Main Effects, Off-diagonal: Interaction Effects)", fontsize=14, weight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("shap_interaction_matrix.pdf", format='pdf', bbox_inches='tight', dpi=1200)
plt.show()

# Save interaction matrix to CSV file
interaction_matrix_test.to_csv("shap_interaction_matrix.csv")
print(f"💾 Interaction matrix has been saved to 'shap_interaction_matrix.csv'")

# Optional: Create an interaction network graph
import networkx as nx

# Create graph object
G = nx.Graph()

# Add nodes and edges
for _, row in interaction_df.iterrows():
    G.add_edge(row['feature1'], row['feature2'], weight=row['interaction_value'])

# Draw network graph
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, k=1, iterations=50)  # Node layout

# Node size based on degree centrality
node_sizes = [G.degree(node) * 500 for node in G.nodes()]

# Edge width based on interaction strength
edge_widths = [G[u][v]['weight'] * 10 for u, v in G.edges()]

# Draw network
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue',
                       alpha=0.9, edgecolors='black')
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

# Add edge weight labels
edge_labels = {(u, v): f"{G[u][v]['weight']:.3f}" for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title('Feature Interaction Network Graph', fontsize=16, weight='bold')
plt.axis('off')  # Turn off axes
plt.tight_layout()
plt.savefig("interaction_network.pdf", format='pdf', bbox_inches='tight', dpi=1200)
plt.show()

# Output interaction statistics
print(f"\n📈 Interaction statistics:")
print(f"Total number of interaction pairs: {len(interaction_df)}")
print(f"Average interaction strength: {interaction_df['interaction_value'].mean():.4f}")
print(f"Maximum interaction strength: {interaction_df['interaction_value'].max():.4f}")
print(f"Minimum interaction strength: {interaction_df['interaction_value'].min():.4f}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set figure parameters
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# First, calculate SHAP interaction values
print("🔄 Calculating SHAP interaction values...")

# Use TreeExplainer to explain the CatBoost model
explainer = shap.TreeExplainer(catboost)

# Calculate SHAP interaction values
shap_interaction_values = explainer.shap_interaction_values(X_test)

print(f"✅ SHAP interaction values calculated!")
print(f"Shape of SHAP interaction values array: {shap_interaction_values.shape}")

# Get feature names
feature_names_test = X_test.columns  # Feature names for the test set

# Initialize an empty DataFrame, size of feature number matrix
interaction_matrix_test = pd.DataFrame(np.nan, index=feature_names_test, columns=feature_names_test)

# Loop through each feature pair to calculate interaction effects
print("📊 Constructing interaction matrix...")
for i, feature in enumerate(feature_names_test):
    for j, other_feature in enumerate(feature_names_test):
        if i != j:  # Only consider interactions between different features
            # Calculate the absolute SHAP interaction values for each sample
            interaction_values = [shap_interaction_values[sample_idx][i, j] for sample_idx in range(shap_interaction_values.shape[0])]
            # Calculate the average absolute interaction value
            avg_interaction_value = np.mean(np.abs(interaction_values))
            # Assign the result to the corresponding position in the matrix
            interaction_matrix_test.loc[feature, other_feature] = avg_interaction_value
        else:
            # Calculate main effect values (on the diagonal)
            main_effect_values = [shap_interaction_values[sample_idx][i, i] for sample_idx in range(shap_interaction_values.shape[0])]
            # Calculate the average absolute main effect value
            avg_main_effect_value = np.mean(np.abs(main_effect_values))
            # Assign the main effect value to the diagonal
            interaction_matrix_test.loc[feature, feature] = avg_main_effect_value

# Output the final interaction matrix
print("✅ Interaction matrix construction complete!")
print("\n📋 Interaction matrix:")
print(interaction_matrix_test)

# Visualize the interaction matrix
plt.figure(figsize=(10, 8))
mask = np.zeros_like(interaction_matrix_test, dtype=bool)
np.fill_diagonal(mask, True)  # Mask the diagonal so it displays differently in the heatmap

# Create heatmap
sns.heatmap(interaction_matrix_test,
            annot=True,
            fmt=".4f",
            cmap="YlOrRd",
            cbar_kws={'label': 'Average Absolute SHAP Value'},
            mask=mask,  # Mask diagonal
            square=True)

plt.title("SHAP Feature Interaction Matrix\n(Diagonal: Main Effects, Off-diagonal: Interaction Effects)", fontsize=14, weight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("shap_interaction_matrix.pdf", format='pdf', bbox_inches='tight', dpi=1200)
plt.show()

# Save interaction matrix to CSV file
interaction_matrix_test.to_csv("shap_interaction_matrix.csv")
print(f"💾 Interaction matrix has been saved to 'shap_interaction_matrix.csv'")

# Optional: Create an interaction network graph
import networkx as nx

# Create graph object
G = nx.Graph()

# Add nodes and edges
for _, row in interaction_df.iterrows():
    G.add_edge(row['feature1'], row['feature2'], weight=row['interaction_value'])

# Draw network graph
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, k=1, iterations=50)  # Node layout

# Node size based on degree centrality
node_sizes = [G.degree(node) * 500 for node in G.nodes()]

# Edge width based on interaction strength
edge_widths = [G[u][v]['weight'] * 10 for u, v in G.edges()]

# Draw network
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue',
                       alpha=0.9, edgecolors='black')
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

# Add edge weight labels
edge_labels = {(u, v): f"{G[u][v]['weight']:.3f}" for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title('Feature Interaction Network Graph', fontsize=16, weight='bold')
plt.axis('off')  # Turn off axes
plt.tight_layout()
plt.savefig("interaction_network.pdf", format='pdf', bbox_inches='tight', dpi=1200)
plt.show()

# Output interaction statistics
print(f"\n📈 Interaction statistics:")
print(f"Total number of interaction pairs: {len(interaction_df)}")
print(f"Average interaction strength: {interaction_df['interaction_value'].mean():.4f}")
print(f"Maximum interaction strength: {interaction_df['interaction_value'].max():.4f}")
print(f"Minimum interaction strength: {interaction_df['interaction_value'].min():.4f}")

import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def plot_feature_interaction_and_main_effects(
        interaction_matrix,
        main_effect_matrix,
        interaction_cmap=cm.Blues,
        main_effect_cmap=cm.Greens,
        title="Feature Interaction and Main Effects Network",
        output_path=None,
        background_circle_color='lightgray'
):
    # Create an empty graph
    G = nx.Graph()

    # Add nodes (features)
    for _, row in main_effect_matrix.iterrows():  # Use main_effect_matrix instead of main_effect_df
        G.add_node(row['feature'], importance=row['main_effect_value'])  # Use main effect value

    # Add edges (interactions)
    for _, row in interaction_matrix.iterrows():
        G.add_edge(row['feature1'], row['feature2'], interaction=row['interaction_value'])

    # Set layout, here using circular layout
    layout_scale = 1.0
    pos = nx.circular_layout(G, scale=layout_scale)

    center_x = 0.0
    center_y = 0.0

    background_circle_radius = layout_scale * 1.0

    # Create main plot and subplot for color bar
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.8, 0.025], height_ratios=[0.5, 0.5])
    gs.update(wspace=-0.4)

    ax_main = fig.add_subplot(gs[:, 0])

    # --- Draw filled background circle (using the added background_circle_color parameter) ---
    circle = plt.Circle((center_x, center_y), background_circle_radius, facecolor=background_circle_color,
                        edgecolor='gray', alpha=0.3, linewidth=0.5, zorder=0)
    ax_main.add_patch(circle)
    # --- End of background circle drawing ---

    # Draw nodes
    node_importance = [G.nodes[node]['importance'] for node in G.nodes()]
    max_importance = max(node_importance)
    min_importance = min(node_importance)

    node_size = [(imp / max_importance) * 2000 + 300 for imp in node_importance]

    node_colors = [main_effect_cmap((imp - min_importance) / (max_importance - min_importance)) for imp in
                   node_importance]

    nodes = nx.draw_networkx_nodes(G, pos, ax=ax_main, node_color=node_colors, node_size=node_size, alpha=1)
    if nodes is not None:
        nodes.set_zorder(3)

    # --- Draw edges (sorted by interaction value, from smallest to largest) ---
    edges_data = []
    for u, v in G.edges():
        edges_data.append((G.edges[u, v]['interaction'], u, v))
    edges_data.sort(key=lambda x: x[0])

    edge_interactions_sorted = [data[0] for data in edges_data]
    edge_list_sorted = [(data[1], data[2]) for data in edges_data]

    max_interaction = max(edge_interactions_sorted)
    min_interaction = min(edge_interactions_sorted)

    edge_width = [(inter / max_interaction) * 5 + 1 for inter in edge_interactions_sorted]
    edge_colors = [interaction_cmap((inter - min_interaction) / (max_interaction - min_interaction)) for inter in
                   edge_interactions_sorted]

    edges = nx.draw_networkx_edges(G, pos, ax=ax_main, edgelist=edge_list_sorted, width=edge_width,
                                   edge_color=edge_colors, alpha=0.7)
    if edges is not None:
        edges.set_zorder(1)

    # Draw node labels
    label_pos = {}
    label_offset = 0.25  # Adjust this value to control the distance of the label from the node

    for node, coords in pos.items():
        x, y = coords
        # Calculate the vector from the center to the node, normalize it, then multiply by the offset
        label_pos[node] = (x * (1 + label_offset), y * (1 + label_offset))

    labels = nx.draw_networkx_labels(G, label_pos, ax=ax_main, font_size=14,
                                     font_weight='bold')  # Node name font size 14, bold
    for text_obj in labels.values():
        text_obj.set_zorder(4)

    ax_main.set_title(title, size=15, fontweight='bold')
    ax_main.axis('off')

    ax_main.set_aspect('equal', adjustable='box')
    ax_main.set_xlim([-background_circle_radius * 1.4, background_circle_radius * 1.4])
    ax_main.set_ylim([-background_circle_radius * 1.4, background_circle_radius * 1.4])

    # Add color bars
    ax_interaction_cbar = fig.add_subplot(gs[0, 1])
    sm_interaction = cm.ScalarMappable(cmap=interaction_cmap)
    sm_interaction.set_array([0, max_interaction])
    cbar_interaction = plt.colorbar(sm_interaction, cax=ax_interaction_cbar, orientation='vertical')
    cbar_interaction.set_label('', rotation=0, labelpad=0, fontsize=12)
    ax_interaction_cbar.set_title('Vint', loc='left', pad=-10, fontsize=15,
                                  fontweight='bold')  # Color bar title font size 15, bold
    cbar_interaction.ax.tick_params(labelsize=13)  # Color bar tick font size 13

    ax_main_effect_cbar = fig.add_subplot(gs[1, 1])
    sm_main_effect = cm.ScalarMappable(cmap=main_effect_cmap)
    sm_main_effect.set_array([0, max_importance])
    cbar_main_effect = plt.colorbar(sm_main_effect, cax=ax_main_effect_cbar, orientation='vertical')
    cbar_main_effect.set_label('', rotation=0, labelpad=0, fontsize=12)
    ax_main_effect_cbar.set_title('MEI', loc='left', pad=-10, fontsize=15,
                                  fontweight='bold')  # Color bar title font size 15, bold
    cbar_main_effect.ax.tick_params(labelsize=13)  # Color bar tick font size 13

    plt.tight_layout()

    if output_path:  # Save the image if a path is provided
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=1200)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

plot_feature_interaction_and_main_effects(
    interaction_matrix=interaction_df,  # Interaction matrix containing interaction values between features
    main_effect_matrix=main_effect_df,  # Main effect matrix containing the main effect values for each feature
    interaction_cmap=cm.Purples,        # Use purple color map for interactions
    main_effect_cmap=cm.Greens,         # Use green color map for main effects
    background_circle_color='lightgray',  # Set background circle color to light gray
    title="Feature Interaction and Main Effects",   # Custom title for the plot
    output_path="Feature Interaction and Main Effects.pdf"     # Save the plot as a PDF file
)

