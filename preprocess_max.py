import pandas as pd
import re
import numpy as np
import os

# New function to extract and max over clusters for "Ours" method
def extract_our_reward_values(file_contents_list):
    all_steps = []
    all_r_values = []

    for file_content in file_contents_list:
        pattern = r'eval.*?S: (\d+)' + ''.join([rf'.*?R_{i}: ([\d\.-]+) \|' for i in range(50)])
        matches = re.findall(pattern, file_content)
        steps = [float(match[0]) for match in matches]  # Ensure steps are float for NaN compatibility
        r_values = [[float(match[i]) for i in range(1, 51)] for match in matches]
        all_steps.append(steps)
        all_r_values.append(r_values)

    max_length = max(len(steps) for steps in all_steps)

    padded_r_values = [np.pad(r_values, ((0, max_length - len(r_values)), (0, 0)), constant_values=np.nan) for r_values in all_r_values]
    padded_steps = [np.pad(np.array(steps, dtype=float), (0, max_length - len(steps)), constant_values=np.nan) for steps in all_steps]

    padded_r_values = np.array(padded_r_values)
    padded_steps = np.array(padded_steps)

    max_r_values = np.nanmax(padded_r_values, axis=0)
    common_steps = padded_steps[0]

    return common_steps, max_r_values

# New function to extract and max over clusters for "Ours" method
def extract_our_success_values(file_contents_list):
    all_steps = []
    all_su_values = []

    for file_content in file_contents_list:
        pattern = r'eval.*?S: (\d+)' + ''.join([rf'.*?Su_{i}: ([\d\.]+)' for i in range(50)])
        matches = re.findall(pattern, file_content)
        steps = [float(match[0]) for match in matches]  # Ensure steps are float for NaN compatibility
        su_values = [[float(match[i]) for i in range(1, 51)] for match in matches]
        all_steps.append(steps)
        all_su_values.append(su_values)

    max_length = max(len(steps) for steps in all_steps)

    padded_su_values = [np.pad(su_values, ((0, max_length - len(su_values)), (0, 0)), constant_values=np.nan) for su_values in all_su_values]
    padded_steps = [np.pad(np.array(steps, dtype=float), (0, max_length - len(steps)), constant_values=np.nan) for steps in all_steps]

    padded_su_values = np.array(padded_su_values)
    padded_steps = np.array(padded_steps)

    max_su_values = np.nanmax(padded_su_values, axis=0)
    common_steps = padded_steps[0]

    return common_steps, max_su_values

# Base directory where the subdirectories are located
base_directory = './Baseline/Ours'
#base_directory = "./ablations/K/K=1"
#base_directory = "./ablations/K/K=2"
#base_directory = "./ablations/K/K=4"
#base_directory = "./ablations/Epsilon/eps=.8_k=3"

# Iterate over clusters and seeds to generate separate CSV files
for cluster in range(3):
   
    for seed in range(3):
        # Paths to "Ours" method files
        #path = os.path.join(base_directory, 'Ours', f'logs.C_{cluster}.{seed}')
        path = os.path.join(base_directory, f'logs.C_{cluster}.{seed}')
        
        # Extracting data for "Ours"
        with open(path, 'r') as file:
            file_content = file.read()

        steps_ours, r_values_ours = extract_our_reward_values([file_content])
        _, su_values_ours = extract_our_success_values([file_content])

        # Creating a DataFrame to save the data
        data_dict = {'Steps': steps_ours}
        for i in range(50):
            data_dict[f'R_{i}'] = r_values_ours[:, i]
            data_dict[f'Su_{i}'] = su_values_ours[:, i]

        df_ours = pd.DataFrame(data_dict)

        # Saving to CSV with cluster and seed in the filename
        csv_filename = f'logs.C_{cluster}.{seed}.csv'
        df_ours.to_csv(csv_filename, index=False)

        print(f"Data saved to {csv_filename}")
import pandas as pd
import os

# Base directory where the generated CSVs are located
csv_directory = './'

# List of cluster and seed combinations
clusters = range(3)
seeds = range(3)

# Iterate over seeds
for seed in seeds:
    # List to hold DataFrames for each cluster
    cluster_dfs = []

    # Load CSVs for each cluster for the current seed
    for cluster in clusters:
        csv_filename = f'logs.C_{cluster}.{seed}.csv'
        csv_path = os.path.join(csv_directory, csv_filename)
        
        # Load CSV into DataFrame
        df = pd.read_csv(csv_path)
        cluster_dfs.append(df)

    # Concatenate all DataFrames for the current seed along the axis of clusters
    concatenated_df = pd.concat(cluster_dfs, axis=1)
    concatenated_df.to_csv(f"intermediate_{seed}.csv")

    # Find the maximum value for each column across clusters
    max_df = concatenated_df.groupby(concatenated_df.columns, axis=1).max()

    # Save the maximum values to a new CSV
    output_filename = f'C_max_{seed}.csv'
    max_df.to_csv(output_filename, index=False)

    print(f"Max values saved to {output_filename}")
import pandas as pd
import os

# Base directory where the max CSVs are located
csv_directory = './'

# List of max CSV filenames
max_filenames = [f'C_max_{i}.csv' for i in range(3)]

# Load all max CSV files into DataFrames
max_dfs = [pd.read_csv(os.path.join(csv_directory, filename)) for filename in max_filenames]

# Initialize lists to hold dataframes for rewards and success
rewards_dfs = []
success_dfs = []

# Separate rewards and success columns
for df in max_dfs:
    rewards_df = df.filter(regex=r'^R_')
    success_df = df.filter(regex=r'^Su_')
    rewards_dfs.append(rewards_df)
    success_dfs.append(success_df)

# Concatenate rewards DataFrames along axis=0 (row-wise) and calculate the mean
rewards_avg_df = sum(rewards_dfs) / len(rewards_dfs)

# Concatenate success DataFrames along axis=0 (row-wise) and calculate the mean directly
success_avg_df = sum(success_dfs) / len(success_dfs)

# Combine the rewards and success DataFrames back together
final_avg_df = pd.concat([rewards_avg_df, success_avg_df], axis=1)

# Save the combined average values to a new CSV
output_filename = 'C_avg_final.csv'
final_avg_df.to_csv(output_filename, index=False)

print(f"Final average values saved to {output_filename}")

# List of files to remove
files_to_remove = [
    'Ours.csv',
    'logs.C_0.0.csv',
    'logs.C_0.1.csv',
    'logs.C_0.2.csv',
    'logs.C_1.0.csv',
    'logs.C_1.1.csv',
    'logs.C_1.2.csv',
    'logs.C_2.0.csv',
    'logs.C_2.1.csv',
    'logs.C_2.2.csv',
    'C_max_0.csv',
    'C_max_1.csv',
    'C_max_2.csv'
]

# Remove each file

for file in files_to_remove:
    try:
        os.remove(file)
        print(f"Removed {file}")
    except FileNotFoundError:
        print(f"File not found: {file}")
    except Exception as e:
        print(f"Error removing {file}: {e}")
