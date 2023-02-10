# NOTE: This file will be responsible for analysing the results of the dataset - mostly deals with formatting
# loading data. The error calculations are not happening here
import csv
import glob
import os
import numpy as np
import pandas as pd
from database import COLMAPDatabase
from parameters import Parameters
from pose_evaluator import pose_evaluate_generic_comparison_model, pose_evaluate_generic_comparison_model_Maa
from query_image import read_images_binary, load_images_from_text_file, get_localised_image_by_names, get_query_images_pose_from_images

def get_cmu_path(dest, slice):
    return os.path.join(dest, slice, "exmaps_data")

def average_all_aggregated_csv_files(dest):
    # always aggerate all here
    slices_names = ["slice2", "slice3", "slice4", "slice5", "slice6", "slice7", "slice8", "slice9", "slice10", "slice11", "slice12", "slice13", "slice14", "slice15",
                    "slice16", "slice17", "slice18", "slice19", "slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
    all_df = pd.DataFrame()
    for slice in slices_names:
        new_df = pd.DataFrame([slice])
        all_df = pd.concat([all_df, new_df], axis=0)
        param_path = get_cmu_path(dest, slice) #slice base path
        parameters = Parameters(param_path)
        slice_df = pd.read_csv(parameters.aggregated_results_csv)
        all_df = pd.concat([all_df, slice_df], axis=0)

    all_df.to_csv(os.path.join(dest, "evaluation_results_2022_all.csv"), index=True, header=True, sep=',')

def get_all_slices_csv_files(dest):
    # always aggerate all here
    slices_names = ["slice2", "slice3", "slice4", "slice5", "slice6", "slice7", "slice8", "slice9", "slice10", "slice11", "slice12", "slice13", "slice14", "slice15",
                    "slice16", "slice17", "slice18", "slice19", "slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
    all_dfs = {}
    for slice in slices_names:
        param_path = get_cmu_path(dest, slice) #slice base path
        parameters = Parameters(param_path)
        slice_df = pd.read_csv(parameters.aggregated_results_csv)
        all_dfs[slice] = slice_df
    return all_dfs

def load_est_poses_results(path):
    # [pose, inliers_no, outliers_no, iterations, elapsed_time]
    return np.load(path, allow_pickle=True).item()

def get_degenerate_poses(est_poses):
    degenerate_poses = []
    for name, values in est_poses.items():
        if est_poses[name][0] is None: #degenerate pose
            degenerate_poses.append(name)
    return degenerate_poses

def count_degenerate_poses(est_poses):
    degenerate_poses_no = 0
    for _, pose_data in est_poses.items():
        if pose_data[0] is None: #degenerate pose
            degenerate_poses_no += 1
            continue
    return degenerate_poses_no

def remove_degenerate_poses(est_poses, degenerate_poses):
    for name in degenerate_poses:
        if name in est_poses.keys():
            del est_poses[name] #no need to return as dicts are pass by reference

# using errors from Benchmarking 6DOF paper (https://www.visuallocalization.net)
def get_6dof_accuracy_for_all_images(est_poses, gt_poses, scale = 1):
    image_errors_6dof = {}
    for image_name, est_values in est_poses.items():
        q_pose = est_values[0] #pose at [0]
        gt_pose = gt_poses[image_name] #only one ground truth
        error_t, error_r = pose_evaluate_generic_comparison_model(q_pose, gt_pose, scale)
        image_errors_6dof[image_name] = [error_t, error_r]
    return image_errors_6dof

# CAB dataset notes (use this in the loop below):
# if (est_values[0] > 90):  # Use this for CAB as it is a very challenging dataset and 1-2 poses return a trans error of 60k+ ! This is out my control.
#     t_errors.append(90)
# else:
#     t_errors.append(est_values[0])

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

    name = os.path.splitext(os.path.basename(result_file))[0][0:-2]
    data_row = [name, total_matches, total_inliers_percentage,
                total_outliers_percentage, iteration_mean, time_mean,
                t_error_mean, r_error_mean, mAA[0], degenerate_poses_no]

    return data_row

def check_for_degenerate_cases(RUNS, results_path):
    degenerate_poses_all = []
    for i in range(RUNS):
        for result_file in glob.glob(os.path.join(results_path, f"*_{i}.npy")):  # each method run (some might have degen some not)
            est_poses_results_all = load_est_poses_results(result_file)
            degenerate_poses_all.extend(get_degenerate_poses(est_poses_results_all))

    for degen in np.unique(degenerate_poses_all):
        print(f"Degenerate case: {degen}")

    return np.unique(degenerate_poses_all)


# This method will read all .csv results files and create a dataframe with the mean results
#  and save to a csv file
def average_csv_results_files(parameters, runs):
    first_run = 0
    # assume the other is the same for all runs
    sample_df = pd.read_csv(os.path.join(parameters.results_path, f"evaluation_results_all_run_{first_run}.csv"))
    methods = sample_df['Method Name']
    values = sample_df.columns[2:]

    all_values = np.empty((len(methods), len(values), runs))
    # reset
    for run in range(runs):
        pf = pd.read_csv(os.path.join(parameters.results_path, f"evaluation_results_all_run_{run}.csv"))
        assert np.all(pf['Method Name'] == methods)
        all_values[:, :, run] = pf.values[:, 2:]

    # Now I have all the values in all_values, matrix, get the mean and
    # add the headers again, methods names and save the file
    all_values_mean = np.nanmean(all_values, axis=2)

    df = pd.DataFrame(all_values_mean, index=methods, columns=values).sort_values(by=['MAA'], ascending=False)
    df.to_csv(parameters.aggregated_results_csv, index=True, header=True, sep=',')

def write_method_results_to_csv(base_path, run, thresholds_q, thresholds_t, degenerate_poses, scale=1):
    print("Base path: " + base_path)
    parameters = Parameters(base_path)
    result_file_output_path = os.path.join(parameters.results_path, f"evaluation_results_all_run_{run}.csv")

    # the "gt" here means ground truth (also used as query) TODO: can move them outside from this method
    query_images_bin_path = os.path.join(parameters.gt_model_images_path)
    query_images_path = os.path.join(parameters.query_images_path)

    query_images = read_images_binary(query_images_bin_path)
    query_images_names = load_images_from_text_file(query_images_path)
    localised_query_images_names = get_localised_image_by_names(query_images_names, query_images)
    query_images_ground_truth_poses_all = get_query_images_pose_from_images(localised_query_images_names, query_images)

    header = ['Method Name', 'Total Matches', 'Inliers (%)', 'Outliers (%)', 'Iterations', 'Total Time (s)', 'Trans Error (m)', 'Rotation Error (d)', 'MAA', 'Degenerate Poses']
    res_df = pd.DataFrame()
    for result_file in glob.glob(os.path.join(parameters.results_path, f"*_{run}.npy")): #all methods from 1 run (NOT ALL RUNS, no averaging happening here)
        est_poses_results_all = load_est_poses_results(result_file) #query poses estimations

        # below I count the degenerate poses for the CURRENT method only
        # called here before I remove the degenerate poses
        degenerate_poses_count = count_degenerate_poses(est_poses_results_all)

        remove_degenerate_poses(est_poses_results_all, degenerate_poses)
        assert len(est_poses_results_all) <= len(query_images_ground_truth_poses_all)

        # mAA = [np.mean(acc), np.array(acc), np.array(acc_q), np.array(acc_t)]
        mAA = pose_evaluate_generic_comparison_model_Maa(est_poses_results_all, query_images_ground_truth_poses_all, thresholds_q, thresholds_t, scale)
        # image_errors_6dof = an array of N by [error_t, error_r]
        image_errors_6dof = get_6dof_accuracy_for_all_images(est_poses_results_all, query_images_ground_truth_poses_all, scale)

        csv_row_data = get_row_data(result_file, mAA, est_poses_results_all, image_errors_6dof, degenerate_poses_count)
        temp_df = pd.DataFrame([csv_row_data])
        res_df = pd.concat([res_df, temp_df], axis=0, ignore_index=True)

    res_df.columns = header
    res_df = res_df.sort_values(by=['Method Name'])
    res_df.to_csv(result_file_output_path, index=True, header=True, sep=',')