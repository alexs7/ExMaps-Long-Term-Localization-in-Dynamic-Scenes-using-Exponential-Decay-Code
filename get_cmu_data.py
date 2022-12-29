# This file will download and create base model a live and a gt for each CMU slice
# Run this on the CYENS machine. You can use python3 here.
import csv
import glob
import os.path
import re
import shutil
import subprocess
import sys
import tarfile
from os.path import abspath
from pathlib import Path
import cv2
import pycolmap
from tqdm import tqdm
import colmap
from analyse_results_helper import average_csv_results_files
from helper import remove_file_safe, remove_folder_safe, empty_points_3D_txt_file, arrange_images_txt_file, arrange_cameras_txt_file, arrange_sessions, \
    create_query_image_names_txt, gen_query_txt, remove_csv_files_from_directory
from parameters import Parameters
from undistort_img import undistort_cmu

dest = abspath(sys.argv[1]) #/media/iNicosiaData/engd_data/cmu
create_all_data = sys.argv[2] == "1" #set to "0" if you just want to generate the results
do_matching = sys.argv[3] #this is for main.py. Set to 0 if you are already have the matches from previous runs

# 16/01/2023 - all combinations (after all temp runs), query can't be 1 as it is base so choose a number from 2-12
combinations = {"slice2": 6, "slice3": 7, "slice4": 4, "slice5": 4, "slice6": 7,
                "slice7": 6, "slice8": 6, "slice9": 3, "slice10": 3, "slice11": 7,
                "slice12": 3, "slice13": 8, "slice14": 10, "slice15": 8, "slice16": 4,
                "slice17": 9, "slice18": 6, "slice19": 6, "slice20": 10, "slice21": 9,
                "slice22": 9, "slice23": 9, "slice24": 7, "slice25": 9 }

# just various runs to make sure no ouliers are found in the results (in translational error and rotational error)
# temp 1 = first run
# combinations = {"slice6": 7, "slice7": 6, "slice8": 6, "slice9": 3, "slice10": 4,
#                 "slice11": 7, "slice12": 5, "slice13": 8, "slice14": 6, "slice15": 8,
#                 "slice16": 10, "slice17": 9, "slice18": 6, "slice19": 6, "slice20": 10,
#                 "slice23": 9, "slice24": 7, "slice25": 9}
#
#
# temp 2 = second run
# combinations = {"slice10": 6, "slice12": 7, "slice13": 9, "slice15": 10, "slice18": 10, "slice19": 10 }

# temp 3 = third run
# combinations = {"slice10": 9, "slice12": 9, "slice13": 11 }

# temp 4 = fourth run (with the new updates in get_visibility_matrix.py)
# combinations = {"slice12": 2}

# temp 5 = fifth run (after the new updates in get_visibility_matrix.py)
# combinations = {"slice16": 4, "slice14": 10, "slice12": 11, "slice10": 3, "slice4": 4}

# temp 6 = sixth run (after the new updates in get_visibility_matrix.py)
# combinations = {"slice12": 3}

if(create_all_data):
    print("Iterating through slices")
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
    print("Applying exp. decay / getting descs avg..")
    for slice in combinations.keys():
        param_path = f"/media/iNicosiaData/engd_data/cmu/{slice}/exmaps_data"

        command = ["python3.8", "get_visibility_matrix.py", param_path]
        subprocess.check_call(command)

        command = ["python3.8", "get_points_3D_mean_descs.py", param_path]
        subprocess.check_call(command)

# At this point and on, I generate results
RUNS = 3 # for benchmarks
for slice in combinations.keys(): #for each slice run RUNS times
    param_path = f"/media/iNicosiaData/engd_data/cmu/{slice}/exmaps_data"
    parameters = Parameters(param_path)

    print("Running evaluators and analysing results..")
    # At this point I will run the main.py to generate results and save them ("analyse_results_models") to the results folder N times
    # Then I will average the results from the N different .csv files
    result_file_output_path = os.path.join(parameters.results_path, "evaluation_results_2022.csv")

    # clean previous evaluation_results .csv files
    remove_csv_files_from_directory(param_path)

    for i in range(RUNS):
        print(f"Run no: {i} ...")
        # 1 is for doing matching (for lamar be careful not to do it twice - just load from disk)
        # 3000, RANSAC and PROSAC iterations

        if (i > 0):  # Do the matching only at the first iteration no need to do it again
            do_matching = "0"

        command = ["python3.8", "main.py", param_path, do_matching, "3000"]
        subprocess.check_call(command)

        command = ["python3.8", "analyse_results_models_cmu.py", param_path]
        subprocess.check_call(command)

        # the result_file is generated from analyse_results_models_cmu.py
        shutil.copyfile(result_file_output_path, os.path.join(param_path, f"evaluation_results_2022_run_{i}.csv"))

    # Now I will average the results from the N (RUNS) different .csv files
    print("Averaging results..")
    average_csv_results_files(param_path, parameters)

# generate one csv for all results
print("Writing all (evaluation_results_2022_aggregated.csv files) results, from each slice, to a single .csv..")
header = ['Method Name', 'Total Matches', 'Inliers (%)', 'Outliers (%)', 'Iterations', 'Total Time (s)', 'Trans Error (m)', 'Rotation Error (d)', 'MAA', 'Degenerate Poses']
result_file_output_path = "/media/iNicosiaData/engd_data/cmu/evaluation_results_2022_all.csv"

with open(result_file_output_path, 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for slice in combinations.keys():
        slice_csv_result_path = f"/media/iNicosiaData/engd_data/cmu/{slice}/exmaps_data/evaluation_results_2022_aggregated.csv"
        with open(slice_csv_result_path, 'r') as slice_result_f:
            reader = csv.reader(slice_result_f)
            data = list(reader)
            data = data[1:]
            sub_header = [slice, "", "", "", "", "", "", "", "", ""]
            writer.writerow(sub_header)
            for row in data:
                writer.writerow(row)
            new_line = ["", "", "", "", "", "", "", "", "", ""]
            writer.writerow(new_line)

print("Done!")