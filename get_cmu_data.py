# This file will download and create base model a live and a gt for each CMU slice
# Run this on the CYENS machine. You can use python3 here.
import glob
import os.path
import pdb
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
import pycolmap
import requests
from tqdm import tqdm
from os.path import abspath
import cv2
import colmap
from helper import remove_file_safe, remove_folder_safe, empty_points_3D_txt_file, arrange_images_txt_file, arrange_cameras_txt_file, remove_folder, arrange_sessions, \
    create_query_image_names_txt, gen_query_txt
from undistort_img import undistort_cmu

dest = abspath(sys.argv[1])

# all combinations
combinations = {"slice2": 6, "slice3": 7, "slice4": 2, "slice5": 4, "slice6": 4,
                "slice7": 2, "slice8": 3, "slice9": 4, "slice10": 2, "slice11": 5,
                "slice12": 3, "slice13": 7, "slice14": 4, "slice15": 5, "slice16": 6,
                "slice17": 7, "slice18": 8, "slice19": 4, "slice20": 9, "slice21": 9,
                "slice22": 9, "slice23": 6, "slice24": 5, "slice25": 5 }

for slice, q_session in combinations.items():
    # stage 1 - Download file
    cmu_slice = slice
    query_session_no = q_session

    os.makedirs(dest, exist_ok=True)
    cmu_url = f"https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Extended-CMU-Seasons/{cmu_slice}.tar"
    dest_file = os.path.join(dest, f"{cmu_slice}.tar")

    # download code - uncomment if you want to download
    # response = requests.get(cmu_url, stream=True)
    # total_size_in_bytes= int(response.headers.get('content-length', 0))
    # block_size = 1024 #1 Kibibyte
    # progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    # with open(dest_file, 'wb') as file:
    #     for data in response.iter_content(block_size):
    #         progress_bar.update(len(data))
    #         file.write(data)
    # progress_bar.close()
    # if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
    #     print("ERROR, something went wrong")
    # !download code

    # stage 2 - Extract file
    cmu_slice_path = os.path.join(dest , cmu_slice)
    cmu_tar = tarfile.open(dest_file)
    cmu_tar.extractall(dest)
    cmu_tar.close()
    exmaps_data_dir = os.path.join(cmu_slice_path, "exmaps_data")
    remove_folder_safe(exmaps_data_dir)

    # stage 3 - Create directories / Remove old ones
    exmaps_data_base_path = os.path.join(exmaps_data_dir, "base")
    base_images_path = os.path.join(exmaps_data_base_path, "images")
    remove_folder_safe(base_images_path)
    exmaps_data_live_path = os.path.join(exmaps_data_dir, "live")
    live_images_path = os.path.join(exmaps_data_live_path, "images")
    remove_folder_safe(live_images_path)
    exmaps_data_gt_path = os.path.join(exmaps_data_dir, "gt")
    gt_images_path = os.path.join(exmaps_data_gt_path, "images")
    remove_folder_safe(gt_images_path)

    # stage 4 - Create sessions
    cmu_sparse_query_images_path = os.path.join(cmu_slice_path, "query")
    sessions_images_path = os.path.join(exmaps_data_dir, "sessions")
    remove_folder_safe(sessions_images_path)
    sessions_path = arrange_sessions(cmu_sparse_query_images_path, sessions_images_path)
    for session_path in tqdm(sessions_path):
        if not any(Path(session_path).iterdir()):
            continue
        session_no = session_path.split("session_")[1]
        if (int(session_no) == query_session_no):
            cp_command = ["cp", "-r", session_path, gt_images_path]
            subprocess.check_call(cp_command)
        else:
            cp_command = ["cp", "-r", session_path, live_images_path]
            subprocess.check_call(cp_command)

    # stage 5 - Undistort images
    db_images_folder = os.path.join(cmu_slice_path, "database")
    db_images = [f for f in os.listdir(db_images_folder) if re.search('c0', f)]

    for img in tqdm(db_images):
        undistorted_img = undistort_cmu(cv2.imread(os.path.join(db_images_folder, img)))
        cv2.imwrite(os.path.join(base_images_path, img), undistorted_img)

    # stage 6 - Extract features from base images
    base_db_path = os.path.join(exmaps_data_base_path, "database.db")
    remove_file_safe(base_db_path)
    colmap.feature_extractor_cmu(base_db_path, base_images_path)

    # stage 7 - Create model with ground truth camera poses
    cmu_sparse = os.path.join(cmu_slice_path, "sparse")
    reconstruction = pycolmap.Reconstruction(cmu_sparse)
    exmaps_data_base_model_path = os.path.join(exmaps_data_base_path, "model")
    remove_folder_safe(exmaps_data_base_model_path)
    reconstruction.write_text(exmaps_data_base_model_path)
    points_3D_file_txt_path = os.path.join(exmaps_data_base_model_path, 'points3D.txt')
    images_file_txt_path = os.path.join(exmaps_data_base_model_path, 'images.txt')
    cameras_file_txt_path = os.path.join(exmaps_data_base_model_path, 'cameras.txt')
    empty_points_3D_txt_file(points_3D_file_txt_path)
    arrange_images_txt_file(base_db_path, db_images, images_file_txt_path)
    arrange_cameras_txt_file(base_db_path, cameras_file_txt_path)

    # stage 8
    colmap.vocab_tree_matcher(base_db_path)

    # stage 9
    colmap.point_triangulator(base_db_path, base_images_path, exmaps_data_base_model_path, exmaps_data_base_model_path)

    # stage 10 - Create live model
    live_db_path = os.path.join(exmaps_data_live_path, "database.db")
    query_live_images_txt_path = os.path.join(exmaps_data_live_path, "query_name.txt")
    create_query_image_names_txt(query_live_images_txt_path, live_images_path)
    shutil.copyfile(base_db_path, live_db_path)
    colmap.feature_extractor_cmu(live_db_path, live_images_path, query_live_images_txt_path, query=True)
    colmap.vocab_tree_matcher(live_db_path, query_live_images_txt_path)
    exmaps_data_live_model_path = os.path.join(exmaps_data_live_path, "model")
    colmap.image_registrator(live_db_path, exmaps_data_base_model_path, exmaps_data_live_model_path)

    # stage 11 - Create query (or gt) model
    gt_db_path = os.path.join(exmaps_data_gt_path, "database.db")
    query_gt_images_txt_path = os.path.join(exmaps_data_gt_path, "query_name.txt")
    create_query_image_names_txt(query_gt_images_txt_path, gt_images_path)
    shutil.copyfile(live_db_path, gt_db_path)
    colmap.feature_extractor_cmu(gt_db_path, gt_images_path, query_gt_images_txt_path, query=True)
    colmap.vocab_tree_matcher(gt_db_path, query_gt_images_txt_path)
    exmaps_data_gt_model_path = os.path.join(exmaps_data_gt_path, "model")
    colmap.image_registrator(gt_db_path, exmaps_data_live_model_path, exmaps_data_gt_model_path)

    # stage 12 - write live sessions metadata
    base_images_no = len(glob.glob1(base_images_path, "*.jpg"))
    gen_query_txt(live_images_path, base_images_no)

    print("----------- Summaries:")

    base_reconstruction = pycolmap.Reconstruction(exmaps_data_base_model_path)
    print(base_reconstruction.summary())
    live_reconstruction = pycolmap.Reconstruction(exmaps_data_live_model_path)
    print(live_reconstruction.summary())
    gt_reconstruction = pycolmap.Reconstruction(exmaps_data_gt_model_path)
    print(gt_reconstruction.summary())

    print("----------- Done!")

# At this point we have all base/live/gt data locally and ready to apply exponential decay.
print("Applying exp. decay / getting descs avg / and main.py (benchmarking)")

slices = ["slice2", "slice3", "slice4", "slice5", "slice6",
                "slice7", "slice8", "slice9", "slice10", "slice11",
                "slice12", "slice13", "slice14", "slice15", "slice16",
                "slice17", "slice18", "slice19", "slice20", "slice21",
                "slice22", "slice23", "slice24", "slice25"]

for slice in slices:
    param_path = f"/media/iNicosiaData/engd_data/cmu/{slice}/exmaps_data"
    command = ["python3", "get_visibility_matrix.py", param_path]
    subprocess.check_call(command)

    command = ["python3", "get_points_3D_mean_descs.py", param_path]
    subprocess.check_call(command)

    command = ["python3", "main.py", param_path, "1"]
    subprocess.check_call(command)

