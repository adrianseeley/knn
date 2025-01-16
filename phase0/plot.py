import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# File names and labels
file_info = [
    ("knn_k_dt_de_average.csv", "Average"),
    ("knn_k_dt_de_linear.csv", "Linear"),
    ("knn_k_dt_de_linear_rooted.csv", "Linear Rooted"),
    ("knn_k_dt_de_reciprocal.csv", "Reciprocal"),
    ("knn_k_dt_de_reciprocal_rooted.csv", "Reciprocal Rooted")
]

# Load all data into a dictionary
data_dict = {}
for file_name, label in file_info:
    data_dict[label] = pd.read_csv(file_name)

# find max CorrectCount in all files
max_correct_count = 0
for label in data_dict:
    max_correct_count = max(max_correct_count, data_dict[label]['CorrectCount'].max())

# Ensure all data has the same K values
unique_k_values = data_dict[file_info[0][1]]['K'].unique()

# Plot configuration
num_rows = len(unique_k_values)
num_cols = len(file_info)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows), dpi=300)

for i, k in enumerate(unique_k_values):
    for j, (file_name, label) in enumerate(file_info):
        #log which step we are on
        print(f"Processing K={k}, {label}")

        subset = data_dict[label][data_dict[label]['K'] == k]
        heatmap_data = subset.pivot(index='DistanceThreshold', columns='DistanceExponent', values='CorrectCount')

        ax = axes[i, j] if num_rows > 1 else axes[j]
        sns.heatmap(
            heatmap_data, 
            annot=False, 
            fmt="d", 
            cmap="coolwarm", 
            cbar_kws={'label': 'CorrectCount'},
            vmin=830,  # Set minimum value for the scale
            vmax=max_correct_count,  # Set maximum value for the scale
            ax=ax
        )
        
        # Title and labels
        ax.set_title(f"K={k}, {label}")
        ax.set_xlabel("Distance Exponent")
        ax.set_ylabel("Distance Threshold")
        
        # Format axis numbers as 0.000
        # ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3f}"))
        # ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3f}"))

# Adjust layout
plt.tight_layout()
output_file = "knn_heatmaps.png"
plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
plt.close()

print(f"Heatmaps saved as {output_file}")
