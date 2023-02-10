# This file is used to print a table for chapter 4 of my thesis
# NOTE: This file reports the average number of matches for all images, and 3D points viewed

import glob
import os
import numpy as np
from scantools.utils.colmap import read_images_binary
from database import COLMAPDatabase
from parameters import Parameters
from query_image import get_image_id, load_images_from_text_file, get_localised_image_by_names_no_tqdm

cmu_folder = "/media/iNicosiaData/engd_data/cmu/"
retail_folder = "/media/iNicosiaData/engd_data/retail_shop/slice1/"
lamar_HGE_folder = "/media/iNicosiaData/engd_data/lamar/HGE_colmap_model/"
lamar_CAB_folder = "/media/iNicosiaData/engd_data/lamar/CAB_colmap_model/"
lamar_LIN_folder = "/media/iNicosiaData/engd_data/lamar/LIN_colmap_model/"

#to get mAA
cmu_csv_file = "/media/iNicosiaData/engd_data/cmu/evaluation_results_2022_all.csv"
retail_csv_file = "/media/iNicosiaData/engd_data/retail_shop/slice1/results/evaluation_results_2022_aggregated.csv"
lamar_HGE_csv_file = "/media/iNicosiaData/engd_data/lamar/HGE_colmap_model/results/evaluation_results_2022_aggregated.csv"
lamar_CAB_csv_file = "/media/iNicosiaData/engd_data/lamar/CAB_colmap_model/results/evaluation_results_2022_aggregated.csv"
lamar_LIN_csv_file = "/media/iNicosiaData/engd_data/lamar/LIN_colmap_model/results/evaluation_results_2022_aggregated.csv"

def last_slice_number(slice_folder):
    return int(slice_folder.split("slice")[-1].split("/")[0])

def get_matches_mean_numbers_from_folder(folder):
    live_matches = np.load(os.path.join(folder, "matches_live.npy"), allow_pickle=True).item()
    base_matches = np.load(os.path.join(folder, "matches_base.npy"), allow_pickle=True).item()

    total_base_matches = []
    for k, v in base_matches.items():
        total_base_matches.append(v.shape[0])
    base_matches_mean = int(np.array(total_base_matches).sum() / len(total_base_matches))

    total_live_matches = []
    for k, v in live_matches.items():
        total_live_matches.append(v.shape[0])
    live_matches_mean = int(np.array(total_live_matches).sum() / len(total_live_matches))

    return base_matches_mean, live_matches_mean

# CMU
cmu_slices_sorted = sorted(glob.glob(os.path.join(cmu_folder, "*/")), key=last_slice_number)

# This method will use the gt image's points3D as the number of points3D seen by the image.
# This makes sense because each point that has not a -1 value in the images.bin files is reprojected back on the image.xys
# So the 3D points that have a value other than -1 are the points that are seen by the image.
# Now I have a base and live map. I have matches of one image (gt) to both maps. I know how many 3D points are seen by that gt image
# from the base and live map. Given that I also know how many 3D points are also seen from the gt map, I can then calculate
# the percentage of 3D points based on the gt map.
# For example if one image sees 40, 3D points in the gt map, and 11 in the base map, and 26 in the live map, then I can write in latex,
# the percentage of how many more 3D points the live map sees compared to the base map. (40 is 100% etc)
# Then I do this for all images, and return the mean percentage for both the base and live map.
def get_matched_base_and_live_3D_points_percentage(base_path):
    parameters = Parameters(base_path)
    db_gt = COLMAPDatabase.connect(parameters.gt_db_path)

    # Note that the gt images where localised in the live map but using COLMAP (my ground truth method).
    # Then I compare the matches of the gt images to the base and live map, using my own trivial matching method, OpenCV's BF matcher.
    all_query_images = read_images_binary(parameters.gt_model_images_path)  # only localised images (but from base,live,gt - we need only gt)
    all_query_images_names = load_images_from_text_file(parameters.query_images_path)  # only gt images (all)
    localised_query_images_names = get_localised_image_by_names_no_tqdm(all_query_images_names, all_query_images)  # only gt images (localised only)

    matched_3D_points_base_per_image = np.load(parameters.points3D_seen_per_image_base, allow_pickle=True).item()
    matched_3D_points_live_per_image = np.load(parameters.points3D_seen_per_image_live, allow_pickle=True).item()

    assert len(localised_query_images_names) == len(matched_3D_points_base_per_image.keys()) == len(matched_3D_points_live_per_image.keys())

    diffs_all_base_percentage = [None] * len(localised_query_images_names)
    diffs_all_live_percentage = [None] * len(localised_query_images_names)
    for i in range(len(localised_query_images_names)):
        image_name = localised_query_images_names[i]
        image_id = int(get_image_id(db_gt, image_name))
        points3D_ids_of_image = all_query_images[image_id].point3D_ids
        all_points_3D_gt_image_sees_no = np.where(points3D_ids_of_image != -1)[0].shape[0]
        base_map_points3D_matched_no = matched_3D_points_base_per_image[image_name]
        live_map_points3D_matched_no = matched_3D_points_live_per_image[image_name]

        base_percentage = base_map_points3D_matched_no * 100 / all_points_3D_gt_image_sees_no
        live_percentage = live_map_points3D_matched_no * 100 / all_points_3D_gt_image_sees_no
        diffs_all_base_percentage[i] = base_percentage
        diffs_all_live_percentage[i] = live_percentage

    # obviously the live map sees more 3D points than the base map
    assert np.mean(diffs_all_live_percentage) > np.mean(diffs_all_base_percentage)
    return int(np.mean(diffs_all_base_percentage)), int(np.mean(diffs_all_live_percentage))

def get_row_latex_string_data(folder, cmu=True):
    base_matches_mean, live_matches_mean = get_matches_mean_numbers_from_folder(folder)
    live_matches_diff = live_matches_mean - base_matches_mean
    if(cmu):
        base_matches_mean_latex = f"& {base_matches_mean}"
    else:
        base_matches_mean_latex = f" {base_matches_mean}" #for other datasets
    live_matches_mean_latex = f"& {live_matches_mean} \\textbf{{(+{live_matches_diff})}}"

    base_map_3D_points_mean_percentage, live_map_3D_points_mean_percentage = get_matched_base_and_live_3D_points_percentage(folder)  # returns ints
    base_map_3D_points_mean_percentage_latex = f"& {base_map_3D_points_mean_percentage}\\%"
    diff_map_3D_points_mean_percentage = live_map_3D_points_mean_percentage - base_map_3D_points_mean_percentage
    live_map_3D_points_mean_percentage_latex = f"& {live_map_3D_points_mean_percentage}\% \\textbf{{(+{diff_map_3D_points_mean_percentage}\\%)}}"

    return base_matches_mean_latex, live_matches_mean_latex, base_map_3D_points_mean_percentage_latex, live_map_3D_points_mean_percentage_latex

def print_start_of_table(label, caption, cmu=True):
    print("\\begin{table} % [h]")
    print("\\centering")
    print("\\caption{\\label{tab:" + label + "}%")
    print(caption)
    print("}\\vspace{0.5em}")
    if(cmu):
        print("\\begin{tabular}{@{}lcccc@{}}")
    else:
        print("\\begin{tabular}{@{}cccc@{}}") # for the other datasets (keep it short)
    print("\\toprule")
    if (cmu):
        print("& Base M. Matches & Live M. Matches &  Base M. 3D points & Live M. 3D points \\\\ \\midrule")
    else:
        print("Base M. Matches & Live M. Matches &  Base M. 3D points & Live M. 3D points \\\\ \\midrule")

def print_end_of_table():
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

print_start_of_table("cmu_all_slices_matches", "TODO: Add description")
for cmu_slice in cmu_slices_sorted:
    exmaps_data_path = os.path.join(cmu_slice, "exmaps_data")

    base_matches_mean_latex, live_matches_mean_latex, base_map_3D_points_mean_percentage_latex, live_map_3D_points_mean_percentage_latex = get_row_latex_string_data(exmaps_data_path)

    cmu_slice_name = cmu_slice.split("/")[-2]
    m_str = cmu_slice_name.split("e")[0] + "e"
    m_no = cmu_slice_name.split("e")[1]
    name = f"\\#{m_no}"

    print(f"{name} {base_matches_mean_latex} {live_matches_mean_latex} {base_map_3D_points_mean_percentage_latex} {live_map_3D_points_mean_percentage_latex} \\\\")
print_end_of_table()

print()

# Retail Shop
print_start_of_table("retail_shop_matches", "TODO: Add description" , cmu=False)
base_matches_mean_latex, live_matches_mean_latex, base_map_3D_points_mean_percentage_latex, live_map_3D_points_mean_percentage_latex = get_row_latex_string_data(retail_folder, cmu=False)
print(f" {base_matches_mean_latex} {live_matches_mean_latex} {base_map_3D_points_mean_percentage_latex} {live_map_3D_points_mean_percentage_latex} \\\\")
print_end_of_table()

print()

# Lamar - HGE
print_start_of_table("lamar_HGE_matches", "TODO: Add description", cmu=False)
base_matches_mean_latex, live_matches_mean_latex, base_map_3D_points_mean_percentage_latex, live_map_3D_points_mean_percentage_latex = get_row_latex_string_data(lamar_HGE_folder, cmu=False)
print(f" {base_matches_mean_latex} {live_matches_mean_latex} {base_map_3D_points_mean_percentage_latex} {live_map_3D_points_mean_percentage_latex} \\\\")
print_end_of_table()

print()

# Lamar - CAB
print_start_of_table("lamar_CAB_matches", "TODO: Add description", cmu=False)
base_matches_mean_latex, live_matches_mean_latex, base_map_3D_points_mean_percentage_latex, live_map_3D_points_mean_percentage_latex = get_row_latex_string_data(lamar_CAB_folder, cmu=False)
print(f" {base_matches_mean_latex} {live_matches_mean_latex} {base_map_3D_points_mean_percentage_latex} {live_map_3D_points_mean_percentage_latex} \\\\")
print_end_of_table()

print()

# Lamar - LIN
print_start_of_table("lamar_LIN_matches", "TODO: Add description", cmu=False)
base_matches_mean_latex, live_matches_mean_latex, base_map_3D_points_mean_percentage_latex, live_map_3D_points_mean_percentage_latex = get_row_latex_string_data(lamar_LIN_folder, cmu=False)
print(f" {base_matches_mean_latex} {live_matches_mean_latex} {base_map_3D_points_mean_percentage_latex} {live_map_3D_points_mean_percentage_latex} \\\\")
print_end_of_table()


