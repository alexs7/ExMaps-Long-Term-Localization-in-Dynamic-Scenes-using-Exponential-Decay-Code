import numpy as np
from pose_evaluator import pose_evaluate
from ransac_comparison import run_comparison

def benchmark(benchmarks_iters, ransac_func, matches, query_images_names, K, query_images_ground_truth_poses, scale, val_idx=None, verbose=False):
    trans_errors_overall = []
    rot_errors_overall = []
    inlers_no = []
    outliers = []
    iterations = []
    time = []

    for i in range(benchmarks_iters):
        if(verbose): print(" Benchmark Iters: " + str(i + 1) + "/" + str(benchmarks_iters), end="\r")
        poses , data = run_comparison(ransac_func, matches, query_images_names, K, val_idx=val_idx)
        trans_errors, rot_errors = pose_evaluate(poses, query_images_ground_truth_poses, scale)

        inliers_mean_total = data.mean(axis=0)[0]
        outliers_mean_total = data.mean(axis=0)[1]
        total_inliers_outliers_mean_perce = inliers_mean_total + outliers_mean_total
        inliers_mean_perce = inliers_mean_total * 100 / total_inliers_outliers_mean_perce
        outliers_mean_perce = outliers_mean_total * 100 / total_inliers_outliers_mean_perce

        # 09/12/2022 - Switched to percentages
        inlers_no.append(inliers_mean_perce)
        outliers.append(outliers_mean_perce)
        iterations.append(data.mean(axis=0)[2])
        time.append(data.mean(axis=0)[3])
        trans_errors_overall.append(np.nanmean(trans_errors))
        rot_errors_overall.append(np.nanmean(rot_errors))

    inlers_no = np.array(inlers_no).mean()
    outliers = np.array(outliers).mean()
    iterations = np.array(iterations).mean()
    time = np.array(time).mean()
    trans_errors_overall = np.array(trans_errors_overall).mean()
    rot_errors_overall = np.array(rot_errors_overall).mean()

    if (verbose): print()
    return inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall

