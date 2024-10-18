from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def read_gain_data(file_path):
    data = {'Gain': []}

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            # skip the header line
            if idx == 0:
                print(f"Skipping header: {line.strip()}")
                continue

            # remove whitespace and split by commas
            parts = line.strip().split(", ")

            try:
                if len(parts) >= 8:
                    # extract the gain value
                    gain_part = parts[8].split(" ")  # split by space
                    gain = float(gain_part[1].strip())  # converted to float
                    data['Gain'].append(gain)
                else:
                    print(f"Line skipped, not enough parts: {parts}")

            except (IndexError, ValueError) as e:
                print(f"Error processing line: {line}. Error: {e}")

    return pd.DataFrame(data)


# File paths
file_path_1 = "extinction_event_ea_task2/group_one_best_results.txt"
file_path_2 = "extinction_event_ea_task2/group_two_best_results.txt"
file_path_3 = "island_ea_task2/group_one_best_results.txt"
file_path_4 = "island_ea_task2/group_two_best_results.txt"

# Reading data
df_1 = read_gain_data(file_path_1)
df_2 = read_gain_data(file_path_2)
df_3 = read_gain_data(file_path_3)
df_4 = read_gain_data(file_path_4)

# Check if DataFrames have data
if df_1.empty or df_2.empty or df_3.empty or df_4.empty:
    print("No valid data found for plotting.")
else:
    # Gain values from all groups
    ea1_gr1 = df_1['Gain']
    ea1_gr2 = df_2['Gain']
    ea2_gr1 = df_3['Gain']
    ea2_gr2 = df_4['Gain']

    # Perform t-tests between the same enemy groups in different EAs
    t_stat_1, p_val_1 = stats.ttest_ind(ea1_gr1, ea2_gr1)  # Group 1 comparison
    t_stat_2, p_val_2 = stats.ttest_ind(ea1_gr2, ea2_gr2)   # Group 2 comparison

    print("T-test enemy group 1: ", stats.ttest_ind(ea1_gr1, ea2_gr1))
    print("T-test enemy group 2: ", stats.ttest_ind(ea1_gr2, ea2_gr2))

    # not sure if relevant
    print("T-test EA1 between groups: ", stats.ttest_ind(ea1_gr1, ea1_gr2))
    print("T-test EA2 between groups: ", stats.ttest_ind(ea2_gr1, ea2_gr2))

    # Create the boxplot with custom colors for the groups
    plt.figure(figsize=(10, 6))
    boxprops_1 = dict(facecolor='skyblue', color='black')  # First color
    boxprops_2 = dict(facecolor='lightgreen', color='black')  # Second color

    # Plotting each group separately to apply different colors
    plt.boxplot(ea1_gr1, positions=[1], patch_artist=True, showmeans=True, boxprops=boxprops_1)
    plt.boxplot(ea1_gr2, positions=[3], patch_artist=True, showmeans=True, boxprops=boxprops_2)
    plt.boxplot(ea2_gr1, positions=[2], patch_artist=True, showmeans=True, boxprops=boxprops_1)
    plt.boxplot(ea2_gr2, positions=[4], patch_artist=True, showmeans=True, boxprops=boxprops_2)

    # x-tick labels for the pairs
    plt.xticks([1, 2, 3, 4], ['Extinction Event EA', 'Island EA','Extinction Event EA','Island EA'], fontsize=12)

    # Add p-values as text on the plot
    plt.text(1.5, min(ea1_gr1.min(), ea2_gr1.min()) - 5, f'p = {p_val_1:.3f}', ha='center', fontsize=12)
    plt.text(3.5, min(ea1_gr2.min(), ea2_gr2.min()) - 5, f'p = {p_val_2:.3f}', ha='center', fontsize=12)

    # Legend for color differentiation
    plt.legend([plt.Line2D([0], [0], color='skyblue', lw=4),
                plt.Line2D([0], [0], color='lightgreen', lw=4)],
               ['Enemy group [1,2,3,7]', 'Enemy group [4,5,6,8]'], loc='best')

    # Background
    plt.gca().set_facecolor('honeydew')  # Light greenish background

    # Grid, labels, and title
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylabel('Gain', fontsize=14)
    plt.title('Distribution of Gains for Extinction Event EA and Island EA\nwith p-values', fontsize=16)

    # Show plot
    plt.tight_layout()
    plt.show()
