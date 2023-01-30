# Arguments
import os
from pathlib import Path
from RANSACParameters import RANSACParameters
from benchmark import benchmark
from database import COLMAPDatabase
from feature_matching_generator import feature_matcher_wrapper
from parameters import Parameters
from point3D_loader import read_points3d_default, get_points3D_xyz_id
import numpy as np
from query_image import read_images_binary, load_images_from_text_file, get_localised_image_by_names, get_query_images_pose_from_images, get_intrinsics_from_camera_bin, \
    get_image_id, save_image_projected_points, read_cameras_binary, get_intrinsics
from ransac_prosac import ransac, ransac_dist, prosac
import sys
import cv2

# 08/12/2022 - base path should contain base, live, gt model
base_path = sys.argv[1] # example: "/home/alex/fullpipeline/colmap_data/CMU_data/slice2/" #trailing "/" or add "exmaps_data"
do_feature_matching = sys.argv[2] == "1" # 1 or 0 do / do not matching
no_iterations = int(sys.argv[3]) # 3000 for all
run = int(sys.argv[4]) # 0,1,2,3,4,5,6,7,8,9 whatever you want

parameters = Parameters(base_path)

print("Doing path: " + base_path)

db_gt = COLMAPDatabase.connect(parameters.gt_db_path) #this database can be used to get the query images descs and ground truth poses for later pose comparison
# Here by "query" I mean the gt images from the gt model - a bit confusing, but think of these images as new incoming images
# that the user sends with his mobile device. Now the intrinsics will have to be picked up from COLMAP as COLMAP changes the focal point.. (bug..)
# If it didn't change them I could have used just the ones extracted from ARCore in the ARCore case, and the ones provided by CMU in the CMU case.
print("Reading gt images .bin (localised)...")
all_query_images = read_images_binary(parameters.gt_model_images_path) #only localised images (but from base,live,gt - we need only gt)
all_query_images_names = load_images_from_text_file(parameters.query_images_path) #only gt images (all)
localised_query_images_names = get_localised_image_by_names(all_query_images_names, all_query_images) #only gt images (localised only)

# Note these are the ground truth query images (not session images) that managed to localise against the LIVE model. Might be a low number.
query_images_names = localised_query_images_names
query_images_ground_truth_poses = get_query_images_pose_from_images(query_images_names, all_query_images)

print("Reading 3D points .bin...")
# the order is different in points3D.txt for some reason COLMAP changes it
points3D_base = read_points3d_default(parameters.base_model_points3D_path)
points3D_xyz_id_base = get_points3D_xyz_id(points3D_base)

points3D_live = read_points3d_default(parameters.live_model_points3D_path)
points3D_xyz_id_live = get_points3D_xyz_id(points3D_live)

cameras_bin = read_cameras_binary(parameters.gt_model_cameras_path)
Ks = get_intrinsics(all_query_images, cameras_bin)

print("Loading descs and 3D points, ids...")
# train_descriptors_base and train_descriptors_live are self-explanatory
# train_descriptors must have the same length as the number of points3D
train_descriptors_base = np.load(parameters.avg_descs_base_path).astype(np.float32)
train_descriptors_live = np.load(parameters.avg_descs_live_path).astype(np.float32)

# points3D_ids, these will be used for sanity checks - they have the same order as the scores (29/12/2022)
live_points_3D_ids = np.load(parameters.live_points_3D_ids_file_path)
base_points_3D_ids = np.load(parameters.base_points_3D_ids_file_path)

# Getting the scores
points3D_per_session_scores = np.load(parameters.per_session_decay_scores_path)
points3D_per_image_scores = np.load(parameters.per_image_decay_scores_path)
points3D_visibility_scores = np.load(parameters.binary_visibility_scores_path)

# normalising the scores
points3D_per_session_scores = points3D_per_session_scores / points3D_per_session_scores.sum()
points3D_per_image_scores = points3D_per_image_scores / points3D_per_image_scores.sum()
points3D_visibility_scores = points3D_visibility_scores / points3D_visibility_scores.sum()

points3D_live_model_scores = [points3D_per_image_scores, points3D_per_session_scores, points3D_visibility_scores] #the order matters - (for later on PROSAC etc, look at ransac_comparison.py)!
# Done getting the scores

# 1: Feature matching

# TIP: Remember we are focusing on the model (and its descs) here so the cases to test are:
# query images , train_descriptors from live model : will match base + query images descs to live_model avg descs -> (this can have multiple cases depending on the points3D score used)
# query images , train_descriptors from base model : will match base + query images descs to base avg descs -> (can only be one case...)

#query descs against base model descs
if(do_feature_matching):
    print("Feature matching...")
    #passing all_query_images - but only use gt images
    matches_base, points3D_seen_per_image_base = feature_matcher_wrapper(db_gt, query_images_names, train_descriptors_base,
                                                                         points3D_xyz_id_base, parameters.ratio_test_val, base_points_3D_ids,
                                                                         all_query_images, points3D_base)
    np.save(parameters.matches_base_save_path, matches_base)
    np.save(parameters.points3D_seen_per_image_base, points3D_seen_per_image_base)
    # passing all_query_images - but only use gt images
    matches_live, points3D_seen_per_image_live = feature_matcher_wrapper(db_gt, query_images_names, train_descriptors_live,
                                                                         points3D_xyz_id_live, parameters.ratio_test_val, live_points_3D_ids,
                                                                         all_query_images, points3D_live, points_scores_array = points3D_live_model_scores)
    np.save(parameters.matches_live_save_path, matches_live)
    np.save(parameters.points3D_seen_per_image_live, points3D_seen_per_image_live)
else:
    print("Skipping feature matching...")
    matches_base = np.load(parameters.matches_base_save_path, allow_pickle=True).item()
    matches_live = np.load(parameters.matches_live_save_path, allow_pickle=True).item()

# 2: Comparisons
print(f"Running benchmark with number of iterations: {no_iterations}")

print(f"Base Model: {RANSACParameters.ransac_base}")
est_poses_results = benchmark(ransac, matches_base, localised_query_images_names, Ks, no_iterations)
np.save(os.path.join(parameters.results_path, f"{RANSACParameters.ransac_base}_{run}.npy"), est_poses_results)

print(f"Base Model: {RANSACParameters.prosac_base}")
est_poses_results = benchmark(prosac, matches_base, localised_query_images_names, Ks, no_iterations, val_idx = RANSACParameters.lowes_distance_inverse_ratio_index)
np.save(os.path.join(parameters.results_path, f"{RANSACParameters.prosac_base}_{run}.npy"), est_poses_results)

# -----

print(f"Live Model: {RANSACParameters.ransac_live}")
est_poses_results = benchmark(ransac, matches_live, localised_query_images_names, Ks, no_iterations)
np.save(os.path.join(parameters.results_path, f"{RANSACParameters.ransac_live}_{run}.npy"), est_poses_results)

print(f"Live Model: {RANSACParameters.ransac_dist_per_image_score}")
est_poses_results = benchmark(ransac, matches_live, localised_query_images_names, Ks, no_iterations, val_idx = RANSACParameters.use_ransac_dist_per_image_score)
np.save(os.path.join(parameters.results_path, f"{RANSACParameters.ransac_dist_per_image_score}_{run}.npy"), est_poses_results)

print(f"Live Model: {RANSACParameters.ransac_dist_per_session_score}")
est_poses_results = benchmark(ransac, matches_live, localised_query_images_names, Ks, no_iterations, val_idx = RANSACParameters.use_ransac_dist_per_session_score)
np.save(os.path.join(parameters.results_path, f"{RANSACParameters.ransac_dist_per_session_score}_{run}.npy"), est_poses_results)

print(f"Live Model: {RANSACParameters.ransac_dist_visibility_score}")
est_poses_results = benchmark(ransac, matches_live, localised_query_images_names, Ks, no_iterations, val_idx = RANSACParameters.use_ransac_dist_visibility_score)
np.save(os.path.join(parameters.results_path, f"{RANSACParameters.ransac_dist_visibility_score}_{run}.npy"), est_poses_results)

for prosac_val_index, prosac_val_name in RANSACParameters.prosac_value_titles.items():
    print(f"Live Model (PROSAC): {prosac_val_name}")
    est_poses_results = benchmark(prosac, matches_live, localised_query_images_names, Ks, no_iterations, val_idx=prosac_val_index)
    np.save(os.path.join(parameters.results_path, f"{prosac_val_name}_{run}.npy"), est_poses_results)

print("Done !")
