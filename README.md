# PACMAN

This is the repo for ICML 2025 accepted paper Learning Policy Committees for Effective Personalization in MDPs with Diverse Tasks (https://arxiv.org/abs/2503.01885)

Our code makes heavy use of https://github.com/niiceMing/CMTA?tab=readme-ov-file 

This method requires generating clusters of metaworld tasks and then learning policies based on each cluster, then after running aggregating the results. As such it is a several step process.



Step 1. Download and install this repo including the specific metaworld version mentioned https://github.com/niiceMing/CMTA?tab=readme-ov-file - this is needed for learning policies.

Step 2. Install the provided FairRL conda env.

Step 3. To get vector representations of tasks run task_2_vec_LLM_50.py. You may wish to change the textural discription if adding or removing tasks. Note you will need to have a huggingface api key configured and exported.

Step 4. Run clustering.py. This will generate clusters for a given K and epsilon.

Step 5. Go to CMTA/config/experiment/metaworld.yml and add your cluster by specifying which tasks are *ignored*. 

Step 6. Run MOORE.sh (provided here). This was our best baseline and it isn't present in the CMTA repo. You need to replace CMTA/mtrl/agent/components/encoder.py with the one present in this directory. You will run this K times, one for each cluster modifying the metaworld.yml file in step 4.

Step 7. Once these 3 are done take the STDOUT of the experiment runs in step 5 and run preprocess_max.py. This will generate a CSV called C_avg_final.csv. This is a raw datafile.

Step 8. Aggregate the data in step 6 with aggregate_results.py. You will need to modify the directory path to the location of C_avg_final.csv. This will write a test and train csv with final results.
