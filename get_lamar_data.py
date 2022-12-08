# Run this on the CYENS machine, requires python 3.8 because of lamar
# The dataset URL is: https://cvg-data.inf.ethz.ch/lamar/
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pycolmap
from scantools.capture import Capture
from scantools import run_capture_to_empty_colmap
from tqdm import tqdm
import colmap
from helper import arrange_images_txt_file_lamar, remove_folder_safe, create_query_image_names_txt, create_query_image_names_txt_lamar

colmap_bin = "colmap"

capture_path = "/media/iNicosiaData/engd_data/lamar/HGE"
base_raw_data_path = "map/raw_data"
base_images_path = os.path.join(capture_path, "sessions", base_raw_data_path)
feature_extraction_template_path = "template_inis/lamar_feature_extraction/colmap_feature_extraction_template.ini"
capture = Capture.load(Path(capture_path))
map_session_key = "map"
session = capture.sessions[map_session_key]  # each session has a unique id

base_path = "/media/iNicosiaData/engd_data/lamar/colmap_model/base"
base_model_path = "/media/iNicosiaData/engd_data/lamar/colmap_model/base/model"
base_triangulated_model_path = "/media/iNicosiaData/engd_data/lamar/colmap_model/base/triangulated_model"
base_cameras_files_path = "/media/iNicosiaData/engd_data/lamar/colmap_model/base/camera_files_extra"
remove_folder_safe(base_path)
remove_folder_safe(base_model_path)
remove_folder_safe(base_cameras_files_path)
remove_folder_safe(base_triangulated_model_path)

run_capture_to_empty_colmap.run(capture, ["map"], Path(base_model_path), ext=".txt")
# I create the "empty" model because I need access to the cameras so I can create the config files for each
# to use in the feature extraction
base_reconstruction = pycolmap.Reconstruction(base_model_path)

# Base Model
base_db_path = os.path.join(base_path, "database.db")

# 1, loop through the cameras, and create the feature extraction files for each camera
for cam_id , camera in tqdm(base_reconstruction.cameras.items()):
    f = open(feature_extraction_template_path, 'r')
    feature_extraction_template = f.read()
    f.close()

    feature_extraction_template = feature_extraction_template.format(camera_model=camera.model_name, camera_params=", ".join(camera.params.astype(str)))
    f = open(os.path.join(base_cameras_files_path, f"colmap_feature_extraction_{cam_id}.ini"), "w")
    f.write(feature_extraction_template)
    f.close()

# 2, loop through the cameras again, and create the feature extraction file
# get the "map" images that belong to that camera
# create an image list txt file
# run the feature extractor with that image list file .txt, and camera params
# Unlike CMU we use SIFT-GPU here due to data volume
base_images_paths = []
for cam_id, camera in tqdm(base_reconstruction.cameras.items()):
    camera_images = {k: v for (k,v) in base_reconstruction.images.items() if v.camera_id == cam_id}
    paths = [str(Path(v.name).relative_to(base_raw_data_path)) for _, v in camera_images.items()]
    assert (len(paths) != 0)
    image_list_path = os.path.join(base_cameras_files_path, f"{cam_id}.txt")
    np.savetxt(image_list_path, paths, fmt="%s")
    for path in paths:
        base_images_paths.append(path)
    colmap.feature_extractor_lamar(base_db_path, base_images_path, cam_id, base_cameras_files_path, image_list_path)


# 3, run the matcher
print("Matching images on the live model")
images_file_txt_path = os.path.join(base_model_path, "images.txt")
arrange_images_txt_file_lamar(base_db_path, base_images_paths, images_file_txt_path) #as per COLMAP FAQ
colmap.vocab_tree_matcher(base_db_path)

# 4, triangulate points
colmap.point_triangulator(base_db_path, images_path, base_model_path, base_triangulated_model_path)

print("Moving on to live model..")

# 5, make live folders
live_path = "/media/iNicosiaData/engd_data/lamar/colmap_model/live"
live_model_path = "/media/iNicosiaData/engd_data/lamar/colmap_model/live/model"
remove_folder_safe(live_path)
remove_folder_safe(live_model_path)

# 6, copy base db to live
live_db_path = os.path.join(live_path, "database.db")
print("Copying base db to live..")
shutil.copyfile(base_db_path, live_db_path)
query_session_phone = capture.sessions["query_phone"] #this contains multiple sessions

run_capture_to_empty_colmap.run(capture, ["query_phone"], Path(live_model_path), ext=".txt")
live_reconstruction = pycolmap.Reconstruction(live_model_path)
# write to binary too, so the image registrator works (will not work with .txt files)
live_reconstruction.write_binary(live_model_path)

live_cameras_files_path = "/media/iNicosiaData/engd_data/lamar/colmap_model/live/camera_files_extra"
remove_folder_safe(live_cameras_files_path)
live_raw_data_path = "query_phone/raw_data"
live_images_path = os.path.join(capture_path, "sessions", live_raw_data_path)

# 7, loop through the cameras, and create the feature extraction files for each camera
for cam_id , camera in tqdm(live_reconstruction.cameras.items()):
    f = open(feature_extraction_template_path, 'r')
    feature_extraction_template = f.read()
    f.close()

    feature_extraction_template = feature_extraction_template.format(camera_model=camera.model_name, camera_params=", ".join(camera.params.astype(str)))
    f = open(os.path.join(live_cameras_files_path, f"colmap_feature_extraction_{cam_id}.ini"), "w")
    f.write(feature_extraction_template)
    f.close()

# 8, extract features
live_images_paths = []
for cam_id, camera in tqdm(live_reconstruction.cameras.items()):
    camera_images = {k: v for (k,v) in live_reconstruction.images.items() if v.camera_id == cam_id}
    paths = [str(Path(v.name).relative_to(live_raw_data_path)) for _, v in camera_images.items()]
    assert (len(paths) != 0)
    image_list_path = os.path.join(live_cameras_files_path, f"{cam_id}.txt")
    np.savetxt(image_list_path, paths, fmt="%s")
    for path in paths:
        live_images_paths.append(path)
    colmap.feature_extractor_lamar(live_db_path, live_images_path, cam_id, live_cameras_files_path, image_list_path)

# 7, create live model, match live images
query_live_images_txt_path = os.path.join(live_path, "query_name.txt")
create_query_image_names_txt_lamar(query_live_images_txt_path, live_images_path)
colmap.vocab_tree_matcher(live_db_path, query_live_images_txt_path)

# 8, register images
colmap.image_registrator(live_db_path, base_model_path, live_model_path)

print("Moving on to gt model..")

# 9, make gt folders
gt_path = "/media/iNicosiaData/engd_data/lamar/colmap_model/gt"
gt_model_path = "/media/iNicosiaData/engd_data/lamar/colmap_model/gt/model"
remove_folder_safe(gt_path)
remove_folder_safe(gt_model_path)

# 10, copy live db to gt
gt_db_path = os.path.join(gt_path, "database.db")
print("Copying live db to gt")
shutil.copyfile(live_db_path, gt_db_path)

query_val_session_phone = capture.sessions["query_val_phone"] #validation queries, they provide gt data
run_capture_to_empty_colmap.run(capture, ["query_val_phone"], Path(gt_model_path), ext=".txt")
gt_reconstruction = pycolmap.Reconstruction(gt_model_path)
# write to binary too, so the image registrator works (will not work with .txt files)
gt_reconstruction.write_binary(gt_model_path)

gt_cameras_files_path = "/media/iNicosiaData/engd_data/lamar/colmap_model/gt/camera_files_extra"
remove_folder_safe(gt_cameras_files_path)
gt_raw_data_path = "query_val_phone/raw_data" #TODO: refactor as the value is duplicated
gt_images_path = os.path.join(capture_path, "sessions", gt_raw_data_path)

# 11, loop through the cameras, and create the feature extraction files for each camera
for cam_id , camera in tqdm(gt_reconstruction.cameras.items()):
    f = open(feature_extraction_template_path, 'r')
    feature_extraction_template = f.read()
    f.close()

    feature_extraction_template = feature_extraction_template.format(camera_model=camera.model_name, camera_params=", ".join(camera.params.astype(str)))
    f = open(os.path.join(gt_cameras_files_path, f"colmap_feature_extraction_{cam_id}.ini"), "w")
    f.write(feature_extraction_template)
    f.close()

# 12, extract features
gt_images_paths = []
for cam_id, camera in tqdm(gt_reconstruction.cameras.items()):
    camera_images = {k: v for (k,v) in gt_reconstruction.images.items() if v.camera_id == cam_id}
    paths = [str(Path(v.name).relative_to(gt_raw_data_path)) for _, v in camera_images.items()]
    assert (len(paths) != 0)
    image_list_path = os.path.join(gt_cameras_files_path, f"{cam_id}.txt")
    np.savetxt(image_list_path, paths, fmt="%s")
    for path in paths:
        gt_images_paths.append(path)
    colmap.feature_extractor_lamar(gt_db_path, gt_images_path, cam_id, gt_cameras_files_path, image_list_path)

# 13, create gt model, match gt images
query_gt_images_txt_path = os.path.join(gt_path, "query_name.txt")
create_query_image_names_txt_lamar(query_gt_images_txt_path, gt_images_path)
colmap.vocab_tree_matcher(gt_db_path, query_gt_images_txt_path)

# 14, register images
colmap.image_registrator(gt_db_path, live_model_path, gt_model_path)

print("Done")
