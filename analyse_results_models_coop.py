# 10/01/2023 - Copied from analyse_results models_cmu.py
# This file will be used to analyse results from main.py
import csv
import glob
import os
import sys
import numpy as np
from analyse_results_helper import load_est_poses_results, clean_degenerate_poses, get_6dof_accuracy_for_all_images, get_row_data
from database import COLMAPDatabase
from parameters import Parameters
from pose_evaluator import pose_evaluate_generic_comparison_model_Maa
from query_image import read_images_binary, load_images_from_text_file, get_localised_image_by_names, get_query_images_pose_from_images

base_path = sys.argv[1]
print("Base path: " + base_path)
parameters = Parameters(base_path)
result_file_output_path = os.path.join(parameters.results_path, "evaluation_results_2022.csv")

print("Reading scale...")
scale = np.load(parameters.ARCORE_scale_path).reshape(1)[0]

print("Loading Data..")
db_gt_path = os.path.join(parameters.gt_db_path)
db_gt = COLMAPDatabase.connect(db_gt_path)  # you need this database to get the query images descs as they do NOT exist in the LIVE db, only in GT db!

# the "gt" here means ground truth (also used as query)
query_images_bin_path = os.path.join(parameters.gt_model_images_path)
query_images_path = os.path.join(parameters.query_images_path)
query_cameras_bin_path = os.path.join(parameters.gt_model_cameras_path)

query_images = read_images_binary(query_images_bin_path)
query_images_names = load_images_from_text_file(query_images_path)
localised_query_images_names = get_localised_image_by_names(query_images_names, query_images)
query_images_ground_truth_poses_all = get_query_images_pose_from_images(localised_query_images_names, query_images)

# for Retail dataset
thresholds_q = np.linspace(0.5, 2, 10)
thresholds_t = np.linspace(0.01, 0.05, 10)

np.set_printoptions(precision=3)
print("Thresholds for Rotation (degrees): " + str(thresholds_q))
print("Thresholds for Translation (meters): " + str(thresholds_t))

print("Writing to .csv..")
header = ['Method Name', 'Total Matches', 'Inliers (%)', 'Outliers (%)', 'Iterations', 'Total Time (s)', 'Trans Error (m)', 'Rotation Error (d)', 'MAA', 'Degenerate Poses']
with open(result_file_output_path, 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for result_file in glob.glob(os.path.join(parameters.results_path, "*.npy")):
        est_poses_results_all = load_est_poses_results(result_file) #query poses estimations

        # below the dicts hold the "clean" poses not degenerates!
        est_poses_results, query_images_ground_truth_poses, degenerate_poses_count = clean_degenerate_poses(est_poses_results_all, query_images_ground_truth_poses_all)

        # mAA = [np.mean(acc), np.array(acc), np.array(acc_q), np.array(acc_t)]
        mAA = pose_evaluate_generic_comparison_model_Maa(est_poses_results, query_images_ground_truth_poses, thresholds_q, thresholds_t, scale)
        # image_errors_6dof = an array of N by [error_t, error_r]
        image_errors_6dof = get_6dof_accuracy_for_all_images(est_poses_results, query_images_ground_truth_poses, scale)

        csv_row_data = get_row_data(result_file, mAA, est_poses_results, image_errors_6dof, degenerate_poses_count)
        writer.writerow(csv_row_data)

print("Done writing to CSV!")
