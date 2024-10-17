from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint


# Function to read gain data from the provided file
def read_gain_data(file_path):
	data = {'Run': [], 'Gain': [], 'Enemy_Group': []}

	with open(file_path, 'r') as f:
		lines = f.readlines()
		for idx, line in enumerate(lines):
			# Skip the header line
			if idx == 0:
				print(f"Skipping header: {line.strip()}")
				continue

			# print(f"Processing line {idx}: {line.strip()}")  # Print each line for debugging

			parts = line.strip().split(", ")
			# print(f"Parsed parts: {parts}")  # Print the parsed parts of the line

			try:
				# Ensure line has the expected format (at least 8 parts)
				if len(parts) >= 8:
					run_part = parts[1].split(" ")
					run = int(run_part[1].strip())  # Extract run number
					gain_part = parts[-1].split(": ")
					gain = float(gain_part[1].strip())  # Extract gain value
					enemy_group_part = parts[2].split(" ")[2].strip()  # Extract enemy group
					# print(f"Run: {run}, Gain: {gain}, Enemy Group: {enemy_group_part}")  # Debug output

					data['Run'].append(run)
					data['Gain'].append(gain)
					data['Enemy_Group'].append(enemy_group_part)
				else:
					print(f"Line skipped, not enough parts: {parts}")
			except (IndexError, ValueError) as e:
				print(f"Error processing line: {line}. Error: {e}")
		# pprint(data)
	return pd.DataFrame(data)


# Read the gain data from the file
file_path = "extinction_event_ea_task2/all_best_individuals_gain_results_for_extinction.txt"
df = read_gain_data(file_path)

# Check if DataFrame has data
if df.empty:
	print("No valid data found for plotting.")
else:
	# Group the data by 'Run' and 'Enemy_Group' and calculate the mean Gain
	average_gain_per_run = df.groupby(['Run', 'Enemy_Group'])['Gain'].mean().reset_index()

	# Separate the groups
	group_one = average_gain_per_run[average_gain_per_run['Enemy_Group'] == 'enemies_group_one']['Gain']
	group_two = average_gain_per_run[average_gain_per_run['Enemy_Group'] == 'enemies_group_two']['Gain']

	pprint(group_one)
	pprint(group_two)

	# Create a boxplot comparing the average Gains for both groups
	plt.figure(figsize=(8, 6))
	plt.boxplot([group_one, group_two], patch_artist=True, labels=['Enemy Group One', 'Enemy Group Two'],
	            showmeans=True, boxprops=dict(facecolor='lightblue'))

	# Adding labels and title
	plt.ylabel('Average Gain', fontsize=14)
	plt.title('Distribution of Average Gains\nfor Enemy Group One vs Enemy Group Two', fontsize=16)

	# Display the boxplot
	plt.grid(True, linestyle='--', alpha=0.7)
	plt.tight_layout()
	plt.show()