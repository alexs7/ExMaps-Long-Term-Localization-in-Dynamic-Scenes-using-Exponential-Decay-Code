# Run this on the CYENS machine
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pycolmap
from scantools.capture import Capture
from scantools import run_capture_to_empty_colmap
from tqdm import tqdm
import colmap
from helper import arrange_images_txt_file_lamar, remove_folder_safe

colmap_bin = "colmap"

capture_path = "/media/iNicosiaData/engd_data/lamar/HGE"
raw_data_path = "map/raw_data"
images_path = os.path.join(capture_path,"sessions", raw_data_path)
feature_extraction_template_path = "template_inis/lamar_feature_extraction/colmap_feature_extraction_template.ini"
capture = Capture.load(Path(capture_path))
map_session_key = "map"
session = capture.sessions[map_session_key]  # each session has a unique id

model_bin_dir = "/media/iNicosiaData/engd_data/lamar/colmap_model"
remove_folder_safe(model_bin_dir)

run_capture_to_empty_colmap.run(capture, ["map"], Path(model_bin_dir), ext=".txt")
reconstruction = pycolmap.Reconstruction(model_bin_dir)

# Base Model
base_db_path = os.path.join(model_bin_dir, "database.db")
base_images_paths = []

# 1, loop through the cameras, and create the feature extraction files for each camera
for cam_id , camera in tqdm(reconstruction.cameras.items()):
    f = open(feature_extraction_template_path, 'r')
    feature_extraction_template = f.read()
    f.close()

    feature_extraction_template = feature_extraction_template.format(camera_model=camera.model_name, camera_params=", ".join(camera.params.astype(str)))
    f = open(os.path.join(model_bin_dir, f"colmap_feature_extraction_{cam_id}.ini"), "w")
    f.write(feature_extraction_template)
    f.close()

# 2, loop through the cameras again, and create the feature extraction file
# get the "map" images that belong to that camera
# create an image list txt file
# run the feature extractor with that image list file .txt, and camera params
# Unlike CMU we use SIFT-GPU here due to data volume
for cam_id, camera in tqdm(reconstruction.cameras.items()):
    camera_images = {k: v for (k,v) in reconstruction.images.items() if v.camera_id == cam_id}
    paths = [str(Path(v.name).relative_to(raw_data_path)) for _, v in camera_images.items()]
    assert (len(paths) != 0)
    image_list_path = os.path.join(model_bin_dir, f"{cam_id}.txt")
    np.savetxt(image_list_path, paths, fmt="%s")
    for path in paths:
        base_images_paths.append(path)
    colmap.feature_extractor_lamar(base_db_path, images_path, cam_id, model_bin_dir, image_list_path)

# 3, run the matcher
# fix camera ids in .txt file
images_file_txt_path = os.path.join(model_bin_dir, "images.txt")
arrange_images_txt_file_lamar(base_db_path, base_images_paths, images_file_txt_path)
import pdb
pdb.set_trace()
# colmap.vocab_tree_matcher(database_path)

# 4, triangulate points
colmap.point_triangulator(base_db_path, images_path, model_bin_dir, model_bin_dir)