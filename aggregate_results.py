import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.size': 20,         # Global font size
    'axes.titlesize': 20,    # Font size of the axes title
    'axes.labelsize': 20,    # Font size of the x and y labels
    'xtick.labelsize': 20,   # Font size of the x tick labels
    'ytick.labelsize': 20,   # Font size of the y tick labels
    'legend.fontsize': 18,   # Font size of the legend
    'figure.titlesize': 20   # Font size of the figure title
})


few= False
# Function to extract the R_i values and the step number (S) from the specific row format
def extract_reward_values(file_content):
    pattern = r'eval.*?S: (\d+)' + ''.join([rf'.*?R_{i}: ([\d\.-]+) \|' for i in range(50)])
    matches = re.findall(pattern, file_content)
    steps = [int(match[0]) for match in matches]
    r_values = [[float(match[i]) for i in range(1, 51)] for match in matches]
    return steps, r_values

# Function to extract the Su_{task} values and the step number (S) from the specific row format
def extract_success_values(file_content):
    pattern = r'eval.*?S: (\d+)' + ''.join([rf'.*?Su_{i}: ([\d\.]+)' for i in range(50)])
    matches = re.findall(pattern, file_content)
    steps = [int(match[0]) for match in matches]
    su_values = [[float(match[i]) for i in range(1, 51)] for match in matches]
    return steps, su_values

# Function to apply a moving average for smoothing
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Function to read files, aggregate data, and compute averages and standard deviations
def read_and_aggregate(files, extract_func):
    all_steps = []
    all_values = []

    for path in files:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        with open(path, 'r') as file:
            content = file.read()
            steps, values = extract_func(content)
            if steps and values:
                all_steps.append(steps)
                all_values.append(values)

    if not all_steps or not all_values:
        print("No data extracted from files.")
        return np.array([]), np.array([]), np.array([])

    max_length = max(len(steps) for steps in all_steps)

    all_steps = np.array([np.pad(np.array(steps, dtype=float), (0, max_length - len(steps)), constant_values=np.nan) for steps in all_steps])
    all_values = np.array([np.pad(np.array(values, dtype=float), ((0, max_length - len(values)), (0, 0)), constant_values=np.nan) for values in all_values])

    num_files = len(files)
    avg_steps = np.nanmean(all_steps, axis=0)
    avg_values = np.nanmean(all_values, axis=0)
    std_values = np.nanstd(all_values, axis=0) / np.sqrt(num_files)

    return avg_steps, avg_values, std_values

# Function to read the new baselines' CSVs and return steps, avg success rate, and std
def read_baseline_csv(path):
    df = pd.read_csv(path)
    steps = df['Steps'].values
    avg_succ_train = df['Average Success rate Train'].values
    avg_succ_test = df['Average Success rate Test'].values
    return steps, avg_succ_train, avg_succ_test

# Process modes
MODES = ["Train", "Test"]

# Initialize storage for the new baselines data
new_baseline_data = {}

# Loop through both train and test modes
for mode in MODES:
    # Define task_ids
    task_ids_in_cluster = list(range(20, 50))  # in cluster (20 to 49)
    task_ids_not_in_cluster = list(range(20))  # not in cluster (0 to 19)
    
    if mode == "Train":
        task_ids_eval = task_ids_in_cluster
    else:
        task_ids_eval = task_ids_not_in_cluster 

    task_ids = task_ids_eval

    # Base directory where the subdirectories are located


    base_directory = <path_to_preprocess_dir>

    # Subdirectories and corresponding files
    files = {
        'CARE': [os.path.join(base_directory, 'CARE', f'CARE_{i}') for i in range(1, 4)],
        #Other basleines go here
        'Ours': [os.path.join(base_directory, 'Ours', 'C_avg_final.csv')],
    }

    # Add new baselines to files
    new_baselines = {
        #Other baselines go here
    }

    # Initialize DataFrames for both rewards and success
    steps_dict = {}
    r_values_list = []
    r_values_std_list = []
    su_values_list = []
    su_values_std_list = []

    # Use steps from one of the other baselines (e.g., CARE)
    baseline_steps = None

    for name, paths in files.items():
        if name == "Ours":
            # Directly load the pre-processed data for "Ours"
            df_ours = pd.read_csv(paths[0])
            avg_r_values = df_ours[[f'R_{i}' for i in range(50)]].values
            avg_su_values = df_ours[[f'Su_{i}' for i in range(50)]].values
            avg_steps = baseline_steps if baseline_steps is not None else np.arange(len(avg_r_values))
        else:
            # Process other baselines
            avg_steps, avg_r_values, std_r_values = read_and_aggregate(paths, extract_reward_values)
            _, avg_su_values, std_su_values = read_and_aggregate(paths, extract_success_values)
            if name == 'CARE':  # Save steps from CARE as the baseline steps
                baseline_steps = avg_steps

        if avg_steps.size > 0 and avg_r_values.size > 0:
            steps_dict[name] = pd.Series(avg_steps)

            for i in range(len(task_ids)):
                r_values_list.append(pd.Series(avg_r_values[:, i], name=f'R_{task_ids[i]}_{name}'))
                if name != "Ours":  # Only append std if it's not "Ours"
                    r_values_std_list.append(pd.Series(std_r_values[:, i], name=f'R_{task_ids[i]}_std_{name}'))

            for i in range(len(task_ids)):
                su_values_list.append(pd.Series(avg_su_values[:, i], name=f'Su_{task_ids[i]}_{name}'))
                if name != "Ours":
                    su_values_std_list.append(pd.Series(std_su_values[:, i], name=f'Su_{task_ids[i]}_std_{name}'))
                else:
                    # Calculate the standard deviation for the binary success ratio for "Ours"
                    su_std = np.sqrt(avg_su_values[:, i] * (1 - avg_su_values[:, i]) / 1)  # Assuming 1 file for "Ours"
                    su_values_std_list.append(pd.Series(su_std, name=f'Su_{task_ids[i]}_std_{name}'))

    # Now handle the new baselines (MAML, PEARL, Varibad, RL2)
    for name, path in new_baselines.items():
        steps, avg_succ_train, avg_succ_test = read_baseline_csv(path[0])
        
        # Calculate the standard deviation using the formula success rate * (1 - success rate) / 3
        std_train = np.sqrt(avg_succ_train * (1 - avg_succ_train) / 3)
        std_test = np.sqrt(avg_succ_test * (1 - avg_succ_test) / 3)
        
        new_baseline_data[name] = {
            "steps": steps,
            "avg_succ_train": avg_succ_train,
            "std_train": std_train,
            "avg_succ_test": avg_succ_test,
            "std_test": std_test
        }

    steps_df = pd.DataFrame(steps_dict)

    r_values_df = pd.concat(r_values_list, axis=1)
    r_values_std_df = pd.concat(r_values_std_list, axis=1)
    su_values_df = pd.concat(su_values_list, axis=1)
    su_values_std_df = pd.concat(su_values_std_list, axis=1)

    k3_columns = ['CARE', 'Ours']

    # Initialize lists to hold data for plotting
    all_sums = {}
    all_ratio_sums = {}

    for name in k3_columns:
        
        max_k3_sums = np.zeros(len(steps_df[name]))
        full_sums = np.zeros(len(steps_df[name]))

        max_k3_sums_std = np.zeros(len(steps_df[name]))
        full_sums_std = np.zeros(len(steps_df[name]))

        ratio_k3_sums = np.zeros(len(steps_df[name]))
        ratio_full_sums = np.zeros(len(steps_df[name]))

        ratio_k3_sums_std = np.zeros(len(steps_df[name]))
        ratio_full_sums_std = np.zeros(len(steps_df[name]))

        for i in task_ids:
            cluster_columns_r = [f'R_{i}_{col}' for col in k3_columns if f'R_{i}_{col}' in r_values_df.columns]
            cluster_std_columns_r = [f'R_{i}_std_{col}' for col in k3_columns if f'R_{i}_std_{col}' in r_values_std_df.columns]

            cluster_columns_su = [f'Su_{i}_{col}' for col in k3_columns if f'Su_{i}_{col}' in su_values_df.columns]
            cluster_std_columns_su = [f'Su_{i}_std_{col}' for col in k3_columns if f'Su_{i}_std_{col}' in su_values_std_df.columns]

            if cluster_columns_r:
                max_k3 = r_values_df[cluster_columns_r].max(axis=1)
                num_maxed_over = len(cluster_columns_r)
                max_k3_std = np.sqrt(np.sum(r_values_std_df[cluster_std_columns_r] ** 2, axis=1)) / np.sqrt(num_maxed_over)

                ratio_k3 = su_values_df[cluster_columns_su].mean(axis=1)
                num_ratio = len(cluster_columns_su)
                ratio_k3_std = np.sqrt(ratio_k3 * (1 - ratio_k3) / num_ratio)

            else:
                max_k3 = np.zeros_like(full_sums)
                max_k3_std = np.zeros_like(full_sums_std)
                ratio_k3 = np.zeros_like(ratio_full_sums)
                ratio_k3_std = np.zeros_like(ratio_full_sums_std)

            if f'R_{i}_{name}' in r_values_df.columns:
                full = r_values_df[f'R_{i}_{name}']
                full_std = r_values_std_df[f'R_{i}_std_{name}'] if f'R_{i}_std_{name}' in r_values_std_df.columns else np.zeros_like(full_sums)
            else:
                full = np.zeros_like(full_sums)
                full_std = np.zeros_like(full_sums_std)

            if f'Su_{i}_{name}' in su_values_df.columns:
                full_su = su_values_df[f'Su_{i}_{name}']
                full_su_std = np.sqrt(full_su * (1 - full_su) / num_ratio)
            else:
                full_su = np.zeros_like(ratio_full_sums)
                full_su_std = np.zeros_like(ratio_full_sums_std)

            max_k3_sums += max_k3
            max_k3_sums_std += max_k3_std
            full_sums += full
            full_sums_std += full_std

            ratio_k3_sums += ratio_k3
            ratio_k3_sums_std += ratio_k3_std
            ratio_full_sums += full_su
            ratio_full_sums_std += full_su_std

        steps_full = steps_df[name]

        smoothed_full_sums = moving_average(full_sums, window_size=20)
        smoothed_full_sums_std = moving_average(full_sums_std, window_size=20)
        smoothed_steps_full = steps_full[19:]  # Adjusting for moving average window

        smoothed_ratio_full_sums = moving_average(ratio_full_sums / len(task_ids), window_size=20)
        smoothed_ratio_full_sums_std = moving_average(ratio_full_sums_std / len(task_ids), window_size=20)

        all_sums[name] = (smoothed_full_sums, smoothed_full_sums_std)
        all_ratio_sums[name] = (smoothed_ratio_full_sums, smoothed_ratio_full_sums_std)
     
    # Color mapping for the different baselines
    colors = {
        'CARE': 'blue',
        'Ours': 'green'

    }

    # Rename "Soft" to "Soft Modulation" in the legend
    legend_names = {
        'CARE': 'CARE',
        'Ours': 'PACMAN'

    }

    def plot_combined(smoothed_steps_full, ylabel, title, filename, data_dict, new_baseline_data=None):
        plt.figure(figsize=(10, 6))
        # Ensure both arrays have the same length after insertion
        smoothed_steps_full = pd.concat([pd.Series([0]), smoothed_steps_full], ignore_index=True)

        # Initialize a list to collect DataFrames
        df_list = []

        for name, (smoothed_full_sums, smoothed_full_sums_std) in data_dict.items():

            # Insert 0 at the start of both the steps and sums arrays
            smoothed_full_sums = np.insert(smoothed_full_sums, 0, 0)
            smoothed_full_sums_std = np.insert(smoothed_full_sums_std, 0, 0)
            


            # Collect into a DataFrame
            df = pd.DataFrame({
                'Steps': smoothed_steps_full,
                'smoothed_full_sums': smoothed_full_sums,
                'smoothed_full_sums_std': smoothed_full_sums_std,
                'Baseline': [name]*len(smoothed_steps_full)
            })
            df_list.append(df)
            if name == "Ours":
                smoothed_full_sums_std=smoothed_full_sums_std/np.sqrt(3)
            # Plot the smoothed data
            plt.plot(smoothed_steps_full, smoothed_full_sums, label=legend_names[name], color=colors[name])
            plt.fill_between(smoothed_steps_full, 
                            np.maximum(smoothed_full_sums - smoothed_full_sums_std, 0), 
                            smoothed_full_sums + smoothed_full_sums_std, 
                            alpha=0.2, color=colors[name])

        # Plot the new baselines if available
        if new_baseline_data:
            #these are for metaRL baselines that didn't go through the process above
            for name, data in new_baseline_data.items():

                # Apply the moving average smoothing to the success rates and std
                smoothed_avg_succ = moving_average(data[f"avg_succ_{mode.lower()}"], window_size=20)
                smoothed_std_succ = moving_average(data[f"std_{mode.lower()}"], window_size=20)

                # Adjust steps to match the smoothed data
                smoothed_steps = data["steps"][19:]
                smoothed_steps = np.insert(smoothed_steps, 0, 0)



                smoothed_std_succ = np.insert(smoothed_std_succ, 0, 0)
                smoothed_avg_succ = np.insert(smoothed_avg_succ, 0, 0)

                # Collect into a DataFrame
                df = pd.DataFrame({
                    'Steps': smoothed_steps,
                    'smoothed_full_sums': smoothed_avg_succ,
                    'smoothed_full_sums_std': smoothed_std_succ,
                    'Baseline': [name]*len(smoothed_steps)
                })
                df_list.append(df)

                # Plot the smoothed success rates and fill for standard deviation
                plt.plot(smoothed_steps, smoothed_avg_succ, label=legend_names[name], color=colors[name])
                plt.fill_between(smoothed_steps, np.maximum(smoothed_avg_succ - smoothed_std_succ, 0), smoothed_avg_succ + smoothed_std_succ, alpha=0.2, color=colors[name])
        if few:
            plt.xlabel('Updates')
        else:
            plt.xlabel('Steps')
        plt.ylabel(ylabel)


        plt.title(title)
        # Conditional legend display
        if few:
            # Conditional placement of the legend
            if True:
                # Move legend outside the plot
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            else:
                # Standard legend inside the plot
                plt.legend()
        # No legend when few is False

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        # Concatenate all DataFrames and print
        if df_list:
            baseline_df = pd.concat(df_list, ignore_index=True)
            df_name = f"Baseline_{mode}_{'few' if few else 'full'}"
            print(f"DataFrame {df_name}:")
            print(baseline_df)
            # Optionally, save to a CSV
            baseline_df.to_csv(f"{df_name}.csv", index=False)
    
    if mode == "Train":
        plot_combined(smoothed_steps_full, 'Rewards', 'Reward Comparison', './train_reward_combined.png', all_sums, new_baseline_data)
        if few:
            plot_combined(smoothed_steps_full, 'Success Ratio', 'Success Ratio Train Tasks', './Train_few.png', all_ratio_sums, new_baseline_data)
        else:
            plot_combined(smoothed_steps_full, 'Success Ratio', 'Success Ratio Train Tasks', './Train.png', all_ratio_sums, new_baseline_data)
    else:
        plot_combined(smoothed_steps_full, 'Rewards', 'Reward Comparison', './test_reward_combined.png', all_sums, new_baseline_data)
        if few:
            plot_combined(smoothed_steps_full, 'Success Ratio', 'Success Ratio Test Tasks', './Test_few.png', all_ratio_sums, new_baseline_data)
        else:
            plot_combined(smoothed_steps_full, 'Success Ratio', 'Success Ratio Test Tasks', './Test.png', all_ratio_sums, new_baseline_data)
    
