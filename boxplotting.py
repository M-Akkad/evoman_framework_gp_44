import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Gain scores for Algorithm 1 and 2 on different enemies
data = {
    'Alg1 Enemy 1': [22, 12, 22, 12, 12],
    'Alg1 Enemy 2': [22, -30, -20, 22, 22],
    'Alg1 Enemy 3': [-30, 22, 22, 22, -20],
    'Alg2 Enemy 1': [90, 100, -30, -60, 100],
    'Alg2 Enemy 2': [88, 84, 82, 88, 78],
    'Alg2 Enemy 3': [-20, 22, 22, 12, -20]
}

# Prepare the data for the boxplot
experiment_names = list(data.keys())
gain_scores = [data[exp] for exp in experiment_names]

# Calculate p-values between Alg1 and Alg2 for each enemy using t-test
p_values = {}
p_values['Enemy 1'] = ttest_ind(data['Alg1 Enemy 1'], data['Alg2 Enemy 1'])[1]
p_values['Enemy 2'] = ttest_ind(data['Alg1 Enemy 2'], data['Alg2 Enemy 2'])[1]
p_values['Enemy 3'] = ttest_ind(data['Alg1 Enemy 3'], data['Alg2 Enemy 3'])[1]

# Plotting the boxplot
plt.figure(figsize=(10, 6))

# Custom colors for the boxplots
colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f']

# Create boxplot with custom colors
box = plt.boxplot(gain_scores, patch_artist=True, labels=experiment_names, showmeans=True, boxprops=dict(facecolor='w'))

# Customize the color of each box
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# Adding standard deviation to the plot using scatter plots
for i, (name, scores) in enumerate(data.items()):
    std_dev = np.std(scores)
    mean_val = np.mean(scores)
    # Plot mean with error bar for standard deviation
    plt.errorbar(i + 1, mean_val, yerr=std_dev, fmt='o', color='black', label='Std Dev' if i == 0 else "")

# Adding labels and title
plt.xlabel('Experiment name')
plt.ylabel('Individual gain')
plt.title('Mean individual gain of best performing individual in Alg1 and Alg2')

# Show p-values as annotations
for i, (enemy, p_value) in enumerate(p_values.items(), 1):
    plt.text(i, max(gain_scores[i-1]) + 5, f'p = {p_value:.3f}', ha='center', fontsize=10)

# Customize the grid to make it similar to the reference image
plt.grid(True, linestyle='--', alpha=0.7)

# Remove top and right borders for a clean look
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Show the plot
plt.tight_layout()
plt.show()
