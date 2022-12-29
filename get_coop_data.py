# This file will download and create base model a live and a gt for the Coop data
# Run this on the CYENS machine. You can use python3 here.
# NOTE: if you want to run them separately then copy paste the commands in terminal
import os
import shutil
import subprocess
import sys
from os.path import abspath
from analyse_results_helper import average_csv_results_files
from helper import remove_csv_files_from_directory
from parameters import Parameters

# For the retail shop dataset I use the same data as I used on weatherwax.
print("Applying exp. decay / getting descs avg / and main.py (benchmarking) / saving results")

param_path = abspath(sys.argv[1])  #/media/iNicosiaData/engd_data/retail_shop/slice1
create_all_data = sys.argv[2] == "1" #set to "0" if you just want to generate the results
do_matching = sys.argv[3] #this is for main.py. Set to 0 if you are already have the matches from previous runs

if(create_all_data):
    command = ["python3.8", "get_visibility_matrix.py", param_path]
    subprocess.check_call(command)

    command = ["python3.8", "get_points_3D_mean_descs.py", param_path]
    subprocess.check_call(command)

parameters = Parameters(param_path)
print("Running evaluators and analysing results..")
# At this point I will run the main.py to generate results and save them ("analyse_results_models") to the results folder N times
# Then I will average the results from the N different .csv files
result_file_output_path = os.path.join(parameters.results_path, "evaluation_results_2022.csv")

# clean previous evaluation_results .csv files
remove_csv_files_from_directory(param_path)

RUNS=3
for i in range(RUNS):
    print(f"Run no: {i} ...")
    # 1 is for doing matching (for lamar be careful not to do it twice - just load from disk)
    # 3000, RANSAC and PROSAC iterations

    if(i > 0): #Do the matching only at the first iteration no need to do it again
        do_matching = "0"

    command = ["python3.8", "main.py", param_path, do_matching, "3000"]
    subprocess.check_call(command)

    command = ["python3.8", "analyse_results_models_coop.py", param_path]
    subprocess.check_call(command)

    # the result_file is generated from analyse_results_models_coop.py
    shutil.copyfile(result_file_output_path, os.path.join(param_path, f"evaluation_results_2022_run_{i}.csv"))

# Now I will average the results from the N different .csv files
print("Averaging results..")
average_csv_results_files(param_path, parameters)

print("Done!")