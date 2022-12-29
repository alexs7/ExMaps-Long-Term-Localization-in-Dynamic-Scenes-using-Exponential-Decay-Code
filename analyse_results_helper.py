# NOTE: This file will be responsible for analysing the results of the dataset - mostly deals with formatting
# loading data. The error calculations are not happening here
import glob
import os
import numpy as np
import pandas as pd
from pose_evaluator import pose_evaluate_generic_comparison_model

def load_est_poses_results(path):
    # [pose, inliers_no, outliers_no, iterations, elapsed_time]
    return np.load(path, allow_pickle=True).item()

def clean_degenerate_poses(est_poses, gt_poses):
    est_poses_cp = {}
    gt_poses_cp = {}
    poses = zip(est_poses, gt_poses)
    degenerate_poses_count = 0
    for est_name, gt_name in poses:
        assert est_name == gt_name
        if est_poses[est_name][0] is None: #degenerate pose
            degenerate_poses_count += 1
            continue
        est_poses_cp[est_name] = est_poses[est_name]
        gt_poses_cp[gt_name] = gt_poses[gt_name]

    return est_poses_cp, gt_poses_cp, degenerate_poses_count

def clean_degenerate_poses_lamar(est_poses, gt_poses):
    est_poses_cp = {}
    gt_poses_cp = {}
    poses = zip(est_poses, gt_poses)
    degenerate_poses_count = 0
    for est_name, gt_name in poses:
        assert est_name == gt_name
        if est_poses[est_name][0] is None: #degenerate pose
            degenerate_poses_count += 1
            continue
        # pose_evaluate_generic_comparison_model() has to be called after the None check otherwise it will fail
        # pose_evaluate_generic_comparison_model() doens not handle None values
        # note the 0 here as we only need the pose not the other data
        error_t, _ = pose_evaluate_generic_comparison_model(est_poses[est_name][0], gt_poses[gt_name])
        # Due to RANSAC and PROSAC randomness some translations are very large.
        # If you ran the same method multiple times you would get different results
        # Instead of running multiple times until I get a "reasonable" I just ignore the results that are too large
        if (error_t > 99):
            degenerate_poses_count += 1
            continue  # This is because RANSAC/PROSAC from before were not able to estimate a pose
        est_poses_cp[est_name] = est_poses[est_name]
        gt_poses_cp[gt_name] = gt_poses[gt_name]

    return est_poses_cp, gt_poses_cp, degenerate_poses_count


# using errors from Benchmarking 6DOF paper (https://www.visuallocalization.net)
def get_6dof_accuracy_for_all_images(est_poses, gt_poses, scale = 1):
    assert len(est_poses) == len(gt_poses)
    image_errors_6dof = {}
    for image_name, est_values in est_poses.items():
        q_pose = est_values[0] #pose at [0]
        gt_pose = gt_poses[image_name] #only one ground truth
        error_t, error_r = pose_evaluate_generic_comparison_model(q_pose, gt_pose, scale)
        image_errors_6dof[image_name] = [error_t, error_r]
    return image_errors_6dof

def get_row_data(result_file, mAA, est_poses_results, image_errors_6dof, degenerate_poses_no):
    # ['Method Name', 'Inliers (%)', 'Outliers (%)', 'Iterations', 'Total Time (s)', 'Trans Error (m)', 'Rotation Error (d)', 'MAA', 'Degenerate Poses'] (and total matches)
    assert len(est_poses_results) == len(image_errors_6dof)
    inliers = [] # %
    outliers = [] # %
    iterations = []
    times = []
    t_errors = []
    r_errors = []
    for image_name, est_values in image_errors_6dof.items():
        # [est_pose, inliers_no, outliers_no, iterations, elapsed_time] = est_poses_results
        # [error_t, error_r] = est_values
        t_errors.append(est_values[0])
        r_errors.append(est_values[1])
        times.append(est_poses_results[image_name][4])
        inliers.append(est_poses_results[image_name][1])
        outliers.append(est_poses_results[image_name][2])
        iterations.append(est_poses_results[image_name][3])

    # data for each method
    t_error_mean = np.mean(t_errors)
    r_error_mean = np.mean(r_errors)
    time_mean = np.mean(times)
    iteration_mean = np.mean(iterations)

    total_inliers = np.sum(inliers)
    total_outliers = np.sum(outliers)
    total_matches = total_inliers + total_outliers
    total_inliers_percentage = 100 * total_inliers / total_matches
    total_outliers_percentage = 100 * total_outliers / total_matches

    name = os.path.splitext(os.path.basename(result_file))[0]
    data_row = [name, total_matches, total_inliers_percentage,
                total_outliers_percentage, iteration_mean, time_mean,
                t_error_mean, r_error_mean, mAA[0], degenerate_poses_no]

    return data_row

# This method will read all .csv results files and create a dataframe with the mean results
#  and save to a csv file
def average_csv_results_files(base_path, parameters):
    methods = 0
    values = 0
    runs = 0
    for csv_file in glob.glob(os.path.join(base_path, "*.csv")):
        if("evaluation_results_2022" in csv_file):
            runs += 1
            with open(csv_file, "r") as f:
                lines = f.readlines()
                if(len(lines) == 0):
                    continue
                methods = len(lines) - 1 #minus header
                values = len(lines[0].split(",")) - 1 #minus method name

    all_values = np.empty((methods, values, runs))
    # reset
    runs = -1
    for csv_file in glob.glob(os.path.join(base_path, "*.csv")):
        if("evaluation_results_2022" in csv_file):
            runs += 1
            with open(csv_file, "r") as f:
                lines = f.readlines()
                if (len(lines) == 0):
                    continue
                for mthd_idx in range(methods):
                    for val_idx in range(values):
                        # mthd_idx + 1, to skip the header
                        # val_idx + 1, to skip the method name
                        all_values[mthd_idx, val_idx, runs] = float(lines[mthd_idx + 1].split(",")[val_idx + 1])

    # Now I have all the values in all_values, matrix, get the mean and
    # add the headers again, methods names and save the file
    all_values_mean = all_values.mean(axis=2)
    for csv_file in glob.glob(os.path.join(base_path, "*.csv")):
        if ("evaluation_results_2022" in csv_file):
            titles = lines[0].strip().split(",")[1:]
            method_names = [method_name.split(",")[0] for method_name in lines[1:]]
            break

    df = pd.DataFrame(all_values_mean, index=method_names, columns=titles).sort_values(by=['MAA'], ascending=False)
    df.to_csv(parameters.aggregated_results_csv, index=True, header=True, sep=',')