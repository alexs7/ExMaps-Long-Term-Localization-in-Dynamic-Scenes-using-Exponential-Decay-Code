# Arguments
import cv2
from pathlib import Path

from RANSACParameters import RANSACParameters
from database import COLMAPDatabase
from feature_matching_generator import feature_matcher_wrapper
from parameters import Parameters
from point3D_loader import read_points3d_default, get_points3D_xyz
import numpy as np
from pose_evaluator import pose_evaluate
from query_image import read_images_binary, load_images_from_text_file, get_localised_image_by_names, \
    get_query_images_pose_from_images, get_intrinsics
from ransac_comparison import run_comparison, sort_matches
from ransac_prosac import ransac, ransac_dist, prosac
import sys

base_path = sys.argv[1] # example: "/home/alex/fullpipeline/colmap_data/CMU_data/slice2/" #trailing "/"
parameters = Parameters(base_path)

print("Doing path: " + base_path)
print()

db_gt = COLMAPDatabase.connect(parameters.gt_db_path) #this database can be used to get the query images descs and ground truth poses for later pose comparison
# Here by "query" I mean the gt images from the gt model - a bit confusing, but think of these images as new new incoming images
# that the user sends with his mobile device. Now the intrinsics will have to be picked up from COLMAP as COLMAP changes the focal point.. (bug..)
# If it didn't change them I could have used just the ones extracted from ARCore in the ARCore case, and the ones provided by CMU in the CMU case.
all_query_images = read_images_binary(parameters.gt_model_images_path)
all_query_images_names = load_images_from_text_file(parameters.query_images_path)
localised_query_images_names = get_localised_image_by_names(all_query_images_names, parameters.gt_model_images_path)

# Note these are the ground truth query images (not session images) that managed to localise against the LIVE model. Might be a low number.
query_images_names = localised_query_images_names
query_images_ground_truth_poses = get_query_images_pose_from_images(query_images_names, all_query_images)

# the order is different in points3D.txt for some reason COLMAP changes it
points3D_base = read_points3d_default(parameters.base_model_points3D_path)
points3D_xyz_base = get_points3D_xyz(points3D_base)

points3D_live = read_points3d_default(parameters.live_model_points3D_path)
points3D_xyz_live = get_points3D_xyz(points3D_live)

scale = 1 # default value
if(Path(parameters.ARCORE_scale_path).is_file()):
    scale = np.loadtxt(parameters.ARCORE_scale_path).reshape(1)[0]

K = get_intrinsics(parameters.gt_model_cameras_path, 3)

# train_descriptors_base and train_descriptors_live are self explanatory
# train_descriptors must have the same length as the number of points3D
train_descriptors_base = np.load(parameters.avg_descs_base_path).astype(np.float32)
train_descriptors_live = np.load(parameters.avg_descs_live_path).astype(np.float32)

# Getting the scores
points3D_reliability_scores_matrix= np.load(parameters.per_image_decay_matrix_path)
points3D_heatmap_vals_matrix = np.load(parameters.per_session_decay_matrix_path)
points3D_visibility_matrix = np.load(parameters.binary_visibility_matrix_path)

points3D_reliability_scores = points3D_reliability_scores_matrix.sum(axis=0)
points3D_heatmap_vals = points3D_heatmap_vals_matrix.sum(axis=0)
points3D_visibility_vals = points3D_visibility_matrix.sum(axis=0)

points3D_reliability_scores = points3D_reliability_scores.reshape([1, points3D_reliability_scores.shape[0]])
points3D_heatmap_vals = points3D_heatmap_vals.reshape([1, points3D_heatmap_vals.shape[0]])
points3D_visibility_vals = points3D_visibility_vals.reshape([1, points3D_visibility_vals.shape[0]])

points3D_reliability_scores = points3D_reliability_scores / points3D_reliability_scores.sum()
points3D_heatmap_vals = points3D_heatmap_vals / points3D_heatmap_vals.sum()
points3D_visibility_vals = points3D_visibility_vals / points3D_visibility_vals.sum()

points3D_live_model_scores = [points3D_heatmap_vals, points3D_reliability_scores, points3D_visibility_vals] #the order matters - (for later on PROSAC etc, look at ransac_comparison.py)!
# Done getting the scores

# 1: Feature matching
print("Feature matching...")

# TIP: Remember we are focusing on the model (and its descs) here so the cases to test are:
# query images , train_descriptors from live model : will match base + query images descs to live_model avg descs -> (this can have multiple cases depending on the points3D score used)
# query images , train_descriptors from base model : will match base + query images descs images descs to base avg descs -> (can only be one case...)

#query descs against base model descs
matches_base = feature_matcher_wrapper(db_gt, query_images_names, train_descriptors_base, points3D_xyz_base, parameters.ratio_test_val, verbose = True)
np.save(parameters.matches_base_save_path, matches_base)
print()
matches_live = feature_matcher_wrapper(db_gt, query_images_names, train_descriptors_live, points3D_xyz_live, parameters.ratio_test_val, verbose = True, points_scores_array = points3D_live_model_scores)
np.save(parameters.matches_live_save_path, matches_live)

matches_base = np.load(parameters.matches_base_save_path, allow_pickle=True).item()
matches_live = np.load(parameters.matches_live_save_path, allow_pickle=True).item()

# Print options
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

# 2: RANSAC Comparison
results = {}

print()
print("Base Model")
print(" RANSAC")
poses , data = run_comparison(ransac, matches_base, query_images_names, K)
trans_errors, rot_errors = pose_evaluate(poses, query_images_ground_truth_poses, scale)
results["ransac_base"] = [poses, data, trans_errors, rot_errors]
print(" Inliers: %1.1f | Outliers: %1.1f | Iterations: %1.1f | Time: %2.2f" % (data.mean(axis=0)[0], data.mean(axis=0)[1], data.mean(axis=0)[2], data.mean(axis=0)[3]))
print(" Trans Error (m): %2.2f | Rotation (Degrees): %2.2f" % (np.nanmean(trans_errors), np.nanmean(rot_errors)))

print()
print(" PROSAC only lowe's ratio - (lowes_distance_inverse_ratio)")
poses , data = run_comparison(prosac, matches_base, query_images_names, K, val_idx= RANSACParameters.lowes_distance_inverse_ratio_index)
trans_errors, rot_errors = pose_evaluate(poses, query_images_ground_truth_poses, scale)
results["prosac_base"] = [poses, data, trans_errors, rot_errors]
print(" Inliers: %1.1f | Outliers: %1.1f | Iterations: %1.1f | Time: %2.2f" % (data.mean(axis=0)[0], data.mean(axis=0)[1], data.mean(axis=0)[2], data.mean(axis=0)[3]))
print(" Trans Error (m): %2.2f | Rotation (Degrees): %2.2f" % (np.nanmean(trans_errors), np.nanmean(rot_errors)))

# -----

print()
print("Live Model")
print(" RANSAC") #No need to run RANSAC multiple times here as it is not using any of the points3D scores
poses , data = run_comparison(ransac, matches_live, query_images_names, K)
trans_errors, rot_errors = pose_evaluate(poses, query_images_ground_truth_poses, scale)
results["ransac_live"] = [poses, data, trans_errors, rot_errors]
print(" Inliers: %1.1f | Outliers: %1.1f | Iterations: %1.1f | Time: %2.2f" % (data.mean(axis=0)[0], data.mean(axis=0)[1], data.mean(axis=0)[2], data.mean(axis=0)[3]))
print(" Trans Error (m): %2.2f | Rotation (Degrees): %2.2f" % (np.nanmean(trans_errors), np.nanmean(rot_errors)))

print()
print(" RANSAC + dist")
poses, data = run_comparison(ransac_dist, matches_live, query_images_names, K, val_idx = RANSACParameters.use_ransac_dist)
trans_errors, rot_errors = pose_evaluate(poses, query_images_ground_truth_poses, scale)
results["ransac_dist_live"] = [poses, data, trans_errors, rot_errors]
print(" Inliers: %1.1f | Outliers: %1.1f | Iterations: %1.1f | Time: %2.2f" % (data.mean(axis=0)[0], data.mean(axis=0)[1], data.mean(axis=0)[2], data.mean(axis=0)[3]))
print(" Trans Error (m): %2.2f | Rotation (Degrees): %2.2f" % (np.nanmean(trans_errors), np.nanmean(rot_errors)))

prosac_value_indices = [ RANSACParameters.lowes_distance_inverse_ratio_index,
                         RANSACParameters.higher_neighbour_val_index,
                         RANSACParameters.higher_neighbour_score_index,
                         RANSACParameters.custom_score_index,
                         RANSACParameters.custom_score_index_2]

print()
print(" PROSAC versions")
np.seterr(divide='ignore', invalid='ignore', over='ignore') # this is because some matches will have a reliability_score of zero. so you might have a division by zero
for prosac_sort_val in prosac_value_indices:
    poses, data = run_comparison(prosac, matches_live, query_images_names, K, val_idx= prosac_sort_val)
    trans_errors, rot_errors = pose_evaluate(poses, query_images_ground_truth_poses, scale)
    results["prosac_live_"+str(prosac_sort_val)] = [poses, data, trans_errors, rot_errors]
    print(RANSACParameters.prosac_value_titles[prosac_sort_val])
    print("  Inliers: %1.1f | Outliers: %1.1f | Iterations: %1.1f | Time: %2.2f" % (data.mean(axis=0)[0], data.mean(axis=0)[1], data.mean(axis=0)[2], data.mean(axis=0)[3]))
    print("  Trans Error (m): %2.2f | Rotation (Degrees): %2.2f" % (np.nanmean(trans_errors), np.nanmean(rot_errors)))

np.save(parameters.save_results_path, results)

print()
print("Done !")
