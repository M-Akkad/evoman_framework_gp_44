import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Gain scores for Normal EA and Island EA on different enemies
data = {
    'Normal 1': [22, 12, 22, 12, 12],
    'Island 1': [90, 100, -30, -60, 100],
    'Normal 2': [22, -30, -20, 22, 22],
    'Island 2': [88, 84, 82, 88, 78],
    'Normal 3': [-30, 22, 22, 22, -20],
    'Island 3': [12, 22, 22, -20, 22]
}

# Prepare the data for the boxplot
experiment_names = list(data.keys())
gain_scores = [data[exp] for exp in experiment_names]

# Calculate p-values between Normal EA and Island EA for each enemy using t-test
p_values = {
    'Enemy 1': ttest_ind(data['Normal 1'], data['Island 1'])[1],
    'Enemy 2': ttest_ind(data['Normal 2'], data['Island 2'])[1],
    'Enemy 3': ttest_ind(data['Normal 3'], data['Island 3'])[1]
}

# Plotting the boxplot
plt.figure(figsize=(12, 8))

# Custom colors for the boxplots: blue for Normal EA, light blue for Island EA
colors = ['#1f77b4', '#aec7e8'] * 3  # Alternating colors for Normal EA and Island EA

# Create boxplot with custom colors
box = plt.boxplot(gain_scores, patch_artist=True, labels=experiment_names, showmeans=True, 
                  boxprops=dict(facecolor='w'), meanprops=dict(marker='o', markerfacecolor='black', markersize=8))

# Customize the color of each box
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# Adding standard deviation to the plot using scatter plots
for i, (name, scores) in enumerate(data.items()):
    std_dev = np.std(scores)
    mean_val = np.mean(scores)
    # Plot mean with error bar for standard deviation
    plt.errorbar(i + 1, mean_val, yerr=std_dev, fmt='o', color='black', capsize=5, label='Std Dev' if i == 0 else "")

# Adding labels and title with larger fonts
plt.xlabel('Experiment name', fontsize=14, fontweight='bold')
plt.ylabel('Individual gain', fontsize=14, fontweight='bold')
plt.title('Individual gain of best performing individual\nin Normal EA and Island EA', fontsize=18)

# Show p-values as annotations with horizontal lines between pairs
for i, (enemy, p_value) in enumerate(p_values.items(), 1):
    x1, x2 = i * 2 - 1, i * 2  # Position of the pair
    y_max = max(max(data[f'Normal {i}']), max(data[f'Island {i}'])) + 10  # Dynamic height based on data
    plt.plot([x1, x2], [y_max, y_max], color='black', lw=1.5)  # Horizontal line
    plt.plot([x1, x1], [y_max, y_max], 'ko')  # Dot at end of line (left)
    plt.plot([x2, x2], [y_max, y_max], 'ko')  # Dot at end of line (right)
    plt.text((x1 + x2) * 0.5, y_max + 5, f'p = {p_value:.3f}', ha='center', fontsize=14)

# Customize the grid to match the reference image
plt.grid(True, linestyle='--', alpha=0.7)

# Remove top and right borders for a clean look
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Set x and y ticks font size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Show the plot
plt.tight_layout()
plt.show()
