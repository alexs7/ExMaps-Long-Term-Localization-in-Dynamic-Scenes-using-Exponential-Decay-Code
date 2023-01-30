# This file will download and create base model a live and a gt for the Coop data
# Run this on the CYENS machine. You can use python3 here.
# NOTE: if you want to run them separately then copy paste the commands in terminal
import glob
import os
import shutil
import subprocess
import sys
from os.path import abspath

import numpy as np

from analyse_results_helper import average_csv_results_files, load_est_poses_results, get_degenerate_poses, write_method_results_to_csv, check_for_degenerate_cases
from helper import remove_csv_files_from_directory, remove_folder_safe
from parameters import Parameters

# For the retail shop dataset I use the same data as I used on weatherwax.
print("Applying exp. decay / getting descs avg / and main.py (benchmarking) / saving results")

param_path = abspath(sys.argv[1])  #/media/iNicosiaData/engd_data/retail_shop/slice1
create_all_data = sys.argv[2] == "1" #set to "0" if you just want to generate the results
do_matching = sys.argv[3] #this is for main.py. Set to 0 if you are already have the matches from previous runs
RUNS = int(sys.argv[4]) #number of runs to average the results
do_main = sys.argv[5] == "1"
do_analysis = sys.argv[6] == "1"

# for Retail dataset
thresholds_q = np.linspace(0.5, 2, 10)
thresholds_t = np.linspace(0.01, 0.05, 10)

np.set_printoptions(precision=3)
print("Thresholds for Rotation (degrees): " + str(thresholds_q))
print("Thresholds for Translation (meters): " + str(thresholds_t))

if(create_all_data):
    command = ["python3.8", "get_visibility_matrix.py", param_path]
    subprocess.check_call(command)

    command = ["python3.8", "get_points_3D_mean_descs.py", param_path]
    subprocess.check_call(command)

parameters = Parameters(param_path)
if(do_main):
    print("Running evaluators and analysing results..")
    # At this point I will run the main.py to generate results and save them N times for each method (N * no of methods)
    # clean previous evaluation_results .csv files
    remove_csv_files_from_directory(param_path)
    remove_folder_safe(parameters.results_path)

    for i in range(RUNS):
        print(f"Run no: {i} ...")
        # 1 is for doing matching (for lamar be careful not to do it twice - just load from disk)
        # 3000, RANSAC and PROSAC iterations

        if(i > 0): #Do the matching only at the first iteration no need to do it again
            do_matching = "0"

        command = ["python3.8", "main.py", param_path, do_matching, "3000", str(i)]
        subprocess.check_call(command)

# Gather degenerate poses from all methods. Each method might have different degenerate poses.
# I accumulated all here and ignore them in the analysis - This is because degen cases happen at random times - out of my control.
print("Checking for degenerate cases..")
degenerate_poses_all = check_for_degenerate_cases(RUNS, parameters.results_path)

# Read from each run the methods .npy and save them into one .csv file (no averaging at this point)
if(do_analysis):
    print("Reading scale...")
    scale = np.load(parameters.ARCORE_scale_path).reshape(1)[0]
    print("Analysing results..")
    for i in range(RUNS):
        print(f"Reading results from run no: {i} ...")
        write_method_results_to_csv(param_path, i, thresholds_q, thresholds_t, degenerate_poses_all, scale)

# Now I will average the results from the N different .csv files
print("Averaging results..")
average_csv_results_files(parameters, RUNS)
print()

print("Done!")