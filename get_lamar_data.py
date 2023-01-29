# Run this on the CYENS machine, requires python 3.8 because of lamar
# The dataset URL is: https://cvg-data.inf.ethz.ch/lamar/

# For LaMAR to use you compiled OpenCV, changed the value here: https://github.com/opencv/opencv/blob/master/modules/features2d/src/matchers.cpp#L763
# to 24 then recomplied and installed it using the following command:
# cmake -D CMAKE_BUILD_TYPE=RELEASE \
#     -D CMAKE_INSTALL_PREFIX=/usr/local \
#     -D INSTALL_C_EXAMPLES=ON \
#     -D INSTALL_PYTHON_EXAMPLES=ON \
#     -D OPENCV_GENERATE_PKGCONFIG=ON \
#     -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
#     -D BUILD_NEW_PYTHON_SUPPORT=ON \
#     -D BUILD_opencv_python3=ON \
#     -D HAVE_opencv_python3=ON \
#     -D PYTHON3_EXECUTABLE=/usr/bin/python3.8 \
#     -D BUILD_EXAMPLES=ON ..
# make -j8
# sudo make install
# then run this to point to the custom version of OpenCV "export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.8/site-packages/"

import os
import shutil
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
import pycolmap
from scantools import run_capture_to_empty_colmap
from scantools.capture import Capture
from tqdm import tqdm
import colmap
from analyse_results_helper import average_csv_results_files
from helper import create_query_image_names_txt_lamar, remove_folder_safe, gen_query_txt_lamar, arrange_images_txt_file_lamar, remove_csv_files_from_directory
from parameters import Parameters
from point3D_loader import read_points3d_default
from query_image import read_images_binary, get_intrinsics_from_camera_bin

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

def project_points_debug(model_path, images_paths, output_folder_path, query_name_images_path = None):
    print("Projecting 3D points on images..")
    model_images = read_images_binary(os.path.join(model_path, "images.bin"))
    model_points3D = read_points3d_default(os.path.join(model_path, "points3D.bin"))
    non_base_images = [] #gt/live images

    if(query_name_images_path != None):
        f = open(query_name_images_path)
        non_base_images = f.readlines()
        non_base_images = [x.strip() for x in non_base_images]
        f.close()

    for img_id, img_data in tqdm(model_images.items()):

        if(len(non_base_images) != 0): #only do live or gt images here
            if((img_data.name in non_base_images) == False):
                continue

        image_path = os.path.join(images_paths, img_data.name)
        image = cv2.imread(image_path)
        for i in range(len(img_data.xys)):
            x = int(img_data.xys[i][0])
            y = int(img_data.xys[i][1])
            center = (x, y)
            cv2.circle(image, center, 8, (255, 0, 0), -1) #it knows the order (x,y) no need to reverse
        output_file_path = os.path.join(output_folder_path, img_data.name.split("/")[-1])
        cv2.imwrite(output_file_path, image)

        pose_r = img_data.qvec2rotmat()
        pose_t = img_data.tvec
        pose = np.c_[pose_r, pose_t]
        Rt = np.r_[pose, [np.array([0, 0, 0, 1])]]
        K = get_intrinsics_from_camera_bin(os.path.join(model_path, "cameras.bin"), img_data.camera_id)
        image = cv2.imread(output_file_path) #load again

        for i in range(len(img_data.point3D_ids)):
            point3D_id = img_data.point3D_ids[i]
            if point3D_id == -1:
                continue
            xy = img_data.xys[i] #just kept for reference
            xyz = np.append(model_points3D[point3D_id].xyz, 1) #add 1 for homogeneous coordinates
            point = K.dot(Rt.dot(xyz)[0:3])
            point = point // point[2]
            x = int(point[0])
            y = int(point[1])
            center = (x, y)
            cv2.circle(image, center, 6, (0, 255, 0), -1)

        cv2.imwrite(output_file_path, image)

def feature_extractor_lamar_wrapper(db_path, f_extraction_template_path, model_images_path,
                                    cameras_files_path, raw_data_path, reconstruction):
    # NOTE: Do not compare the ids from 'reconstruction' with the ids in the database. You create the db.
    # The authors created the 'reconstruction' object from whatever. So the ids are NOT the same.

    images_paths = []
    # iphone only images !
    ios_images = {k: v for (k, v) in reconstruction.images.items() if ("ios" in v.name)}
    print(f"Images size: {len(reconstruction.images.items())}")
    print(f"Images subset size: {len(ios_images.items())}")

    for cam_id, camera in tqdm(reconstruction.cameras.items()):
        # read
        f = open(f_extraction_template_path, 'r')
        feature_extraction_template = f.read()
        f.close()
        # write
        feature_extraction_template = feature_extraction_template.format(camera_model=camera.model_name,
                                                                         camera_params=", ".join(camera.params.astype(str)))
        f = open(os.path.join(cameras_files_path, f"colmap_feature_extraction_{cam_id}.ini"), "w")
        f.write(feature_extraction_template)
        f.close()

        # get the "map" images that belong to that camera
        # create an image list txt file
        # run the feature extractor with that image list file .txt, and camera params
        # Unlike CMU we use SIFT-GPU here due to data volume
        camera_images = {k: v for (k, v) in ios_images.items() if (v.camera_id == cam_id)}

        if(len(camera_images) == 0): #because I only use the ios images, cim_id might have only images from hololens (thus empty)
            continue

        # image paths relative to "raw_data_path", i.e. 'ios_2022-07-03_16.00.37_000/images/151298057352.jpg'
        # relative ones (to model_images_path)
        paths = [str(Path(v.name).relative_to(raw_data_path)) for _, v in camera_images.items()]
        assert (len(paths) != 0)

        image_list_path = os.path.join(cameras_files_path, f"{cam_id}.txt")
        np.savetxt(image_list_path, paths, fmt="%s") #tell to the extractor which images to process
        colmap.feature_extractor_lamar(db_path, model_images_path, cam_id, cameras_files_path, image_list_path)

        # return the images relative paths, to create the empty images.txt file for triangulation
        for path in paths:
            images_paths.append(path)
    return images_paths

colmap_bin = "colmap"

param_path = sys.argv[1] #"/media/iNicosiaData/engd_data/lamar/HGE_colmap_model/"
capture_path = sys.argv[2] #"/media/iNicosiaData/engd_data/lamar/HGE"
parameters = Parameters(param_path)
create_all_data = sys.argv[3] == "1" #set to "0" if you just want to generate the results
do_matching = sys.argv[4] #this is for main.py. Set to 0 if you are already have the matches from previous runs

if(create_all_data):
    # remove_folder_safe(param_path)
    # remove_folder_safe(parameters.debug_images_base_path)
    # capture_sessions = os.path.join(capture_path, "sessions")
    #
    # base_raw_data_path = "map/raw_data"
    # base_images_path = os.path.join(capture_path, "sessions", base_raw_data_path)
    feature_extraction_template_path = "template_inis/lamar_feature_extraction/colmap_feature_extraction_template.ini"
    #
    capture = Capture.load(Path(capture_path))
    # map_session_key = "map"
    # session = capture.sessions[map_session_key]  # each session has a unique id
    #
    # base_path = os.path.join(param_path, "base")
    base_model_path = os.path.join(param_path, "base/model")
    base_triangulated_model_path = os.path.join(param_path, "base/triangulated_model")
    # base_cameras_files_path = os.path.join(param_path, "base/camera_files_extra")
    # remove_folder_safe(base_path)
    # remove_folder_safe(base_model_path)
    # remove_folder_safe(base_cameras_files_path)
    # remove_folder_safe(base_triangulated_model_path)
    #
    # run_capture_to_empty_colmap.run(capture, ["map"], Path(base_model_path), ext=".txt") #GT poses in metric
    # # I create the "empty" model because I need access to the cameras so I can create
    # # the config files for each to use in the feature extraction
    # base_reconstruction = pycolmap.Reconstruction(base_model_path)
    #
    # # Base Model
    # base_db_path = os.path.join(base_path, "database.db")
    #
    # # 1, feature extractor wrapper - returns a list of images (relative path)
    # base_images_paths = feature_extractor_lamar_wrapper(base_db_path, feature_extraction_template_path, base_images_path,
    #                                                     base_cameras_files_path, base_raw_data_path, base_reconstruction)
    #
    # # 2, run the matcher
    # print("Matching images on the live model")
    # images_file_txt_path = os.path.join(base_model_path, "images.txt")
    # # write to images_file_txt_path the images you want in the base model (subset)
    # # The method below will create an empty almost .txt file that the triangulator will use for poses
    # arrange_images_txt_file_lamar(base_db_path, base_images_paths, images_file_txt_path) #as per COLMAP FAQ
    # colmap.vocab_tree_matcher(base_db_path)
    #
    # # 3, triangulate points
    # colmap.point_triangulator(base_db_path, base_images_path, base_model_path, base_triangulated_model_path)
    # project_points_debug(base_triangulated_model_path, base_images_path, parameters.debug_images_base_path)
    #
    # print("Moving on to live model..")
    #
    # # 4, make live folders
    live_path = os.path.join(param_path, "live")
    live_model_path = os.path.join(param_path, "live/model")
    live_registered_model_path = os.path.join(param_path, "live/registered_model")
    # remove_folder_safe(live_path)
    # remove_folder_safe(live_model_path)
    # remove_folder_safe(live_registered_model_path)
    # remove_folder_safe(parameters.debug_images_live_path)
    #
    # # 5, copy base db to live
    live_db_path = os.path.join(live_path, "database.db")
    # print("Copying base db to live..")
    # shutil.copyfile(base_db_path, live_db_path)
    # query_session_phone = capture.sessions["query_phone"] #this contains multiple sessions
    #
    # run_capture_to_empty_colmap.run(capture, ["query_phone"], Path(live_model_path), ext=".txt")
    live_reconstruction = pycolmap.Reconstruction(live_model_path)
    #
    live_cameras_files_path = os.path.join(param_path, "live/camera_files_extra")
    # # remove_folder_safe(live_cameras_files_path)
    live_raw_data_path = "query_phone/raw_data"
    live_images_files_path = os.path.join(capture_path, "sessions", live_raw_data_path)

    # 6, live feature extractor
    feature_extractor_lamar_wrapper(live_db_path, feature_extraction_template_path, live_images_files_path,
                                    live_cameras_files_path, live_raw_data_path, live_reconstruction)

    # 7, create live model, match live images
    query_live_images_txt_path = os.path.join(live_path, "query_name.txt")
    create_query_image_names_txt_lamar(query_live_images_txt_path, live_images_files_path) #for matching, not sure if used though
    colmap.vocab_tree_matcher(live_db_path, query_live_images_txt_path)

    # 8, register images
    colmap.image_registrator(live_db_path, base_triangulated_model_path, live_registered_model_path)
    project_points_debug(live_registered_model_path, live_images_files_path, parameters.debug_images_live_path, query_live_images_txt_path)

    # 9, create session lengths file (txt) - used in get_visibility_matrix.py
    base_images_no = len(read_images_binary(os.path.join(base_triangulated_model_path, "images.bin")))
    gen_query_txt_lamar(live_path, live_images_files_path, base_images_no)

    print("Moving on to gt model..")

    # 10, make gt folders
    gt_path = os.path.join(param_path, "gt")
    gt_model_path = os.path.join(param_path, "gt/model")
    gt_registered_model_path = os.path.join(param_path, "gt/registered_model")
    remove_folder_safe(gt_path)
    remove_folder_safe(gt_model_path)
    remove_folder_safe(gt_registered_model_path)
    remove_folder_safe(parameters.debug_images_gt_path)

    # 11, copy live db to gt
    gt_db_path = os.path.join(gt_path, "database.db")
    print("Copying live db to gt")
    shutil.copyfile(live_db_path, gt_db_path)

    query_val_session_phone = capture.sessions["query_val_phone"] #validation queries, they provide gt data
    run_capture_to_empty_colmap.run(capture, ["query_val_phone"], Path(gt_model_path), ext=".txt")
    gt_reconstruction = pycolmap.Reconstruction(gt_model_path)

    gt_cameras_files_path = os.path.join(param_path, "gt/camera_files_extra")
    remove_folder_safe(gt_cameras_files_path)
    gt_raw_data_path = "query_val_phone/raw_data"
    gt_images_files_path = os.path.join(capture_path, "sessions", gt_raw_data_path)

    # 12, loop through the cameras, and create the feature extraction files for each camera
    feature_extractor_lamar_wrapper(gt_db_path, feature_extraction_template_path, gt_images_files_path,
                                    gt_cameras_files_path, gt_raw_data_path, gt_reconstruction)

    # 13, create gt model, match gt images
    query_gt_images_txt_path = os.path.join(gt_path, "query_name.txt")

    create_query_image_names_txt_lamar(query_gt_images_txt_path, gt_images_files_path)
    colmap.vocab_tree_matcher(gt_db_path, query_gt_images_txt_path)

    # 14, register images
    colmap.image_registrator(gt_db_path, live_registered_model_path, gt_registered_model_path)
    project_points_debug(gt_registered_model_path, gt_images_files_path, parameters.debug_images_gt_path, query_gt_images_txt_path)

    print("Done!")

    print("Applying exp. decay / getting descs avg / and main.py (benchmarking) / saving results")

    # For Lamar you need to copy the models to the correct folder
    print("Copying folders first.. (Need to add code)")
    # Move the required files FROM base/triangulated, live/registered_model etc, (*.bin) TO the base/model, live/model, gt/model folders
    remove_folder_safe(base_model_path)
    remove_folder_safe(live_model_path)
    remove_folder_safe(gt_model_path)
    shutil.copytree(base_triangulated_model_path, base_model_path, dirs_exist_ok=True)
    shutil.copytree(live_registered_model_path, live_model_path, dirs_exist_ok=True)
    shutil.copytree(gt_registered_model_path, gt_model_path, dirs_exist_ok=True)

    command = ["python3.8", "get_visibility_matrix.py", param_path]
    subprocess.check_call(command)

    command = ["python3.8", "get_points_3D_mean_descs.py", param_path]
    subprocess.check_call(command)

print("Running evaluators and analysing results..")

# At this point I will run the main.py to generate results and save them ("analyse_results_models") to the results folder N times
# Then I will average the results from the N different .csv files
result_file_output_path = os.path.join(parameters.results_path, "evaluation_results_2022.csv")

# clean previous evaluation_results .csv files
remove_csv_files_from_directory(param_path)

RUNS=3
for i in range(RUNS):
    print(f"Run no: {i} ...")
    # 1 is for doing matching (for lamar be careful not to do it twice - just load from disk)
    # 10000, RANSAC and PROSAC iterations

    if(i > 0): #Do the matching only at the first iteration no need to do it again
        do_matching = "0"

    command = ["python3.8", "main.py", param_path, do_matching, "10000"]
    subprocess.check_call(command)

    command = ["python3.8", "analyse_results_models_lamar.py", param_path]
    subprocess.check_call(command)

    # the result_file is generated from analyse_results_models_lamar.py
    shutil.copyfile(result_file_output_path, os.path.join(param_path, f"evaluation_results_2022_run_{i}.csv"))

# Now I will average the results from the N (RUNS) different .csv files
print("Averaging results..")
average_csv_results_files(param_path, parameters)

print("Done!")