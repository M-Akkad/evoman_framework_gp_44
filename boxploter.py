from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint

import sns


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
					# extract the run number (assuming 'Run x')
					gain_part = parts[8].split(" ")  # split by space
					gain = float(gain_part[1].strip())  # converted to integer
					data['Gain'].append(gain)
				else:
					print(f"Line skipped, not enough parts: {parts}")

			except (IndexError, ValueError) as e:
				print(f"Error processing line: {line}. Error: {e}")

	return pd.DataFrame(data)


file_path_1 = "extinction_event_ea_task2/group_one_best_results.txt"
file_path_2 = "extinction_event_ea_task2/group_two_best_results.txt"
file_path_3 = "island_ea_task2/group_one_best_results.txt"
file_path_4 = "island_ea_task2/group_two_best_results.txt"
df_1 = read_gain_data(file_path_1)
df_2 = read_gain_data(file_path_2)
df_3 = read_gain_data(file_path_3)
df_4 = read_gain_data(file_path_4)


# check DataFrame has data
if df_1.empty or df_2.empty or df_3.empty or df_4.empty:
	print("No valid data found for plotting.")
else:
	# Gain values from all groups
	group_one = df_1['Gain']
	group_two = df_2['Gain']
	group_three = df_3['Gain']
	group_four = df_4['Gain']

	plt.figure(figsize=(10, 6))

	# create the boxplot with custom colors for the groups
	boxprops_1 = dict(facecolor='skyblue', color='black')  # First color
	boxprops_2 = dict(facecolor='lightgreen', color='black')  # Second color

	# Plotting each group separately to apply different colors
	plt.boxplot(group_one, positions=[1], patch_artist=True, showmeans=True, boxprops=boxprops_1)
	plt.boxplot(group_two, positions=[2], patch_artist=True, showmeans=True, boxprops=boxprops_2)
	plt.boxplot(group_three, positions=[3], patch_artist=True, showmeans=True, boxprops=boxprops_1)
	plt.boxplot(group_four, positions=[4], patch_artist=True, showmeans=True, boxprops=boxprops_2)

	# x-tick labels for the pairs
	plt.xticks([1.5, 3.5], ['EA1', 'EA2'], fontsize=12)

	# legend for color differentiation
	plt.legend([plt.Line2D([0], [0], color='skyblue', lw=4),
	            plt.Line2D([0], [0], color='lightgreen', lw=4)],
	           ['Enemy group [1,2,3,7]', 'Enemy group [4,5,6,8]'], loc='best')

	# background
	plt.gca().set_facecolor('honeydew')  # Light greenish background

	# grid, labels, and title
	plt.grid(True, linestyle='--', alpha=0.5)
	plt.ylabel('Gain', fontsize=14)
	plt.title('Distribution of Gains for EA1 and EA2', fontsize=16)

	plt.tight_layout()
	plt.show()