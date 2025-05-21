# PACMAN

This is the repo for ICML 2025 accepted paper Learning Policy Committees for Effective Personalization in MDPs with Diverse Tasks (https://arxiv.org/abs/2503.01885)

*YOU MUST READ CAREFULLY OR YOU WILL HAVE A BAD TIME.*

Our code makes heavy use of https://github.com/niiceMing/CMTA?tab=readme-ov-file 

This method requires generating clusters of metaworld tasks and then learning policies based on each cluster, then after running aggregating the results. As such it is a several step process.



Step 1. Download and install  https://github.com/niiceMing/CMTA?tab=readme-ov-file including the specific metaworld version mentioned- this is needed for learning policies in the paper. If you are learning non metaworld or policies some other way this is not needed. 


Step 2. Install the provided FairRL conda env.


Step 3. You can skip this step if you want to use our embeddings located in the directory to this repo. To get vector representations of tasks run task_2_vec_LLM_50.py. You may wish to change the textural discription if adding or removing tasks. Note you will need to have a huggingface api key configured and exported.


Step 4. If you wish to use our clustering you can see the resulting files in metaworld_c0_nrand.yaml,metaworld_c1_nrand.yaml, and metaworld_c2_nrand.yaml for metaworld. Otherwise run clustering.py. This will generate clusters for a given K and epsilon. clustering.py assumes the embeddings are in ./content (this was originally ran in google colab)


Step 5. Go to CMTA/config/experiment/metaworld.yml and add your cluster by specifying which tasks are *ignored*. The provided metaworld.yml has all tasks removed by default. Our .yml files used in the paper are named metaworld_c0_nrand.yaml,metaworld_c1_nrand.yaml, and metaworld_c2_nrand.yaml. Simply comment the task with a hash (#) if its apart of your cluster. You will have K of these metaworld.yml files for this method. They must be named metaworld.yml and be in that location when the script below is called. 


Step 6. Run MOORE.sh (provided here, we provide care.sh for reference but all these files are in the repo from step 1). MOORE was our best baseline and it isn't present in the CMTA repo. You need to replace CMTA/mtrl/agent/components/encoder.py with the one present in this directory. You will run this K times, one for each cluster modifying the metaworld.yml file in step 4.

Step 7. Once these 3 are done take the STDOUT of the experiment runs in step 5 and run preprocess_max.py. This will generate a CSV called C_avg_final.csv. This is a raw datafile.

Step 8. Aggregate the data in step 6 with aggregate_results.py. You will need to modify the directory path to the location of C_avg_final.csv. This will write a test and train csv with final results.







Note on checkpointing. We have should_resume = True in our metaworld.yml. This is different from the CMTA repo. This should allow you to simply rerun MOORE.sh should it fail midstream for whatever reason, and then simply paste the outputs together prior to step 7.
