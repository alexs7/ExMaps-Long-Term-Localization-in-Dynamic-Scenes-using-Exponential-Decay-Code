# This file is used to print a table for chapter 4 of my thesis
# NOTE: you will need to add \clearpage in the .tex file manually
# You will need to review the table Latex code manually to make sure it is correct
# add caption, labels etc.

# TODO: Add thousands separator to numbers
# TODO: Add the name of the top score method in the table

import sys
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from RANSACParameters import RANSACParameters
from analyse_results_helper import get_all_slices_csv_files
from latex_helper_methods import get_methods_from_slice_cmu, return_data_from_row, get_methods_from_slice, print_start_of_table, latex_dict_for_methods, \
    print_end_of_table, print_all_rows, append_to_metric_tables, print_specific_row, print_start_of_metric_table_aggregated_metrics, get_latex_row_string_top_methods, clear_page, get_base_live_top_score_methods, print_aggregated_table_row

print(">>>>>>>>>>>>>>>>>>> Copy from below this line (for APPENDIX) <<<<<<<<<<<<<<<<<<<<<<<<<<<")
# The tables below will contain all the methods for all methods , sorted my mAA

cmu_csv_dir = "/media/iNicosiaData/engd_data/cmu/"
retail_csv_file = "/media/iNicosiaData/engd_data/retail_shop/slice1/results/evaluation_results_2022_aggregated.csv"
lamar_HGE_csv_file = "/media/iNicosiaData/engd_data/lamar/HGE_colmap_model/results/evaluation_results_2022_aggregated.csv"
lamar_CAB_csv_file = "/media/iNicosiaData/engd_data/lamar/CAB_colmap_model/results/evaluation_results_2022_aggregated.csv"
lamar_LIN_csv_file = "/media/iNicosiaData/engd_data/lamar/LIN_colmap_model/results/evaluation_results_2022_aggregated.csv"

# CMU
df = get_all_slices_csv_files(cmu_csv_dir)
for slice_name, sub_frame in df.items():
    slice_no = slice_name.split("e")[-1]
    desc = f"CMU Slice \\#{slice_no} results, showing the total matches, inliers and outliers percentage, iterations, time in milliseconds, translation, rotation " \
           "error and mean average accuracy (mAA) for all methods. The best performing method is highlighted in bold (by mAA). The values were obtained by averaging all " \
           "the metrics over all the query images in the slice, and then averaging over 5 benchmark runs."
    base_pd, live_pd = get_methods_from_slice_cmu(sub_frame, RANSACParameters.methods_to_evaluate)
    print_start_of_table(label= f"{slice_name}_all_results", caption=desc)

    print_all_rows(base_pd, live_pd, f"Slice \\#{slice_no}")
    print_end_of_table()

    if("11" in slice_name): #This is because of latex (it breaks after too many floats)
        clear_page()

clear_page()

# Retail
retail_df = pd.read_csv(retail_csv_file)
desc = f"Retail shop results, showing the total matches, inliers and outliers percentage, iterations, time in milliseconds, translation, rotation " \
       "error and mean average accuracy (mAA) for all methods. The best performing method is highlighted in bold (by mAA). The values were obtained by averaging all " \
       "the metrics over all the query images in the slice, and then averaging over 5 benchmark runs."
print_start_of_table(label= f"retail_shop_results", caption=desc)
title = "Retail Shop"
base_pd, live_pd = get_methods_from_slice(retail_df, RANSACParameters.methods_to_evaluate)
print_all_rows(base_pd, live_pd, title)
print_end_of_table()

clear_page()

# Lamar - HGE
lamar_HGE_df = pd.read_csv(lamar_HGE_csv_file)
desc = f"LaMAR HGE results, showing the total matches, inliers and outliers percentage, iterations, time in milliseconds, translation, rotation " \
       "error and mean average accuracy (mAA) for all methods. The best performing method is highlighted in bold (by mAA). The values were obtained by averaging all " \
       "the metrics over all the query images in the slice, and then averaging over 5 benchmark runs."
print_start_of_table(label= f"lamar_HGE_results", caption=desc)
title = "LaMAR-HGE"
base_pd, live_pd = get_methods_from_slice(lamar_HGE_df, RANSACParameters.methods_to_evaluate)
print_all_rows(base_pd, live_pd, title)
print_end_of_table()

# Lamar - CAB
lamar_CAB_df = pd.read_csv(lamar_CAB_csv_file)
desc = f"LaMAR CAB results, showing the total matches, inliers and outliers percentage, iterations, time in milliseconds, translation, rotation " \
       "error and mean average accuracy (mAA) for all methods. The best performing method is highlighted in bold (by mAA). The values were obtained by averaging all " \
       "the metrics over all the query images in the slice, and then averaging over 5 benchmark runs."
print_start_of_table(label= f"lamar_CAB_results", caption=desc)
title = "LaMAR-CAB"
base_pd, live_pd = get_methods_from_slice(lamar_CAB_df, RANSACParameters.methods_to_evaluate)
print_all_rows(base_pd, live_pd, title)
print_end_of_table()

# Lamar - LIN
lamar_LIN_df = pd.read_csv(lamar_LIN_csv_file)
desc = f"LaMAR LIN results, showing the total matches, inliers and outliers percentage, iterations, time in milliseconds, translation, rotation " \
       "error and mean average accuracy (mAA) for all methods. The best performing method is highlighted in bold (by mAA). The values were obtained by averaging all " \
       "the metrics over all the query images in the slice, and then averaging over 5 benchmark runs."
print_start_of_table(label= f"lamar_LIN_results", caption=desc)
title = "LaMAR-LIN"
base_pd, live_pd = get_methods_from_slice(lamar_LIN_df, RANSACParameters.methods_to_evaluate)
print_all_rows(base_pd, live_pd, title)
print_end_of_table()

clear_page()

print(">>>>>>>>>>>>>>>>>>> Now printing base, live, and the top score method. (for main thesis - not used for now (05/02/2023) <<<<<<<<<<<<<<<<<<<<<<<<<<<")

# At this point you can print the ranked methods (base, live and the 1 top ranking with score)

# CMU
all_top_score_method = defaultdict(list) #for CMU only
all_top_score_method_occurrence = [] #for CMU only
all_slice_metrics = {}
for slice_name, sub_frame in df.items():
    base_pd, live_pd = get_methods_from_slice_cmu(sub_frame, RANSACParameters.methods_to_evaluate)
    print_start_of_table(label=f"{slice_name}_top_methods", caption="TODO: Add description")
    ransac_base_row, ransac_live_row, top_score_row = get_base_live_top_score_methods(base_pd, live_pd)
    print_specific_row(ransac_base_row, ransac_base_row["Method Name"])
    print_specific_row(ransac_live_row, ransac_live_row["Method Name"])
    print_specific_row(top_score_row, top_score_row["Method Name"])
    print_end_of_table()
    # collecting CMU data
    all_slice_metrics[slice_name] = { "base" : return_data_from_row(ransac_base_row),
                                      "live" : return_data_from_row(ransac_live_row),
                                      "top_score_method" : return_data_from_row(top_score_row)}
    all_top_score_method[top_score_row["Method Name"]].append(top_score_row['MAA'])
    all_top_score_method_occurrence.append(top_score_row["Method Name"])

print("Top performing methods for CMU:")
print("Occurrence")
print(Counter(all_top_score_method_occurrence).most_common())
sorted_all_top_score_method = {}
for method_name, maa_list in all_top_score_method.items():
    sorted_all_top_score_method[method_name] = np.mean(maa_list)
sorted_all_top_score_method = {k: v for k, v in sorted(sorted_all_top_score_method.items(), key=lambda item: item[1], reverse=True)}
print("Mean MAA values")
print(sorted_all_top_score_method)
print()

# Retail
retail_df = pd.read_csv(retail_csv_file)
print_start_of_table(label= f"retail_shop_top_methods", caption="TODO: Add description")
base_pd, live_pd = get_methods_from_slice(retail_df, RANSACParameters.methods_to_evaluate)
ransac_base_row, ransac_live_row, top_score_row = get_base_live_top_score_methods(base_pd, live_pd)
print_specific_row(ransac_base_row, ransac_base_row["Method Name"])
print_specific_row(ransac_live_row, ransac_live_row["Method Name"])
print_specific_row(top_score_row, top_score_row["Method Name"])
print_end_of_table()
all_top_score_method[top_score_row["Method Name"]].append(top_score_row['MAA'])

# Lamar - HGE
lamar_HGE_df = pd.read_csv(lamar_HGE_csv_file)
print_start_of_table(label= f"lamar_HGE_top_methods", caption="TODO: Add description")
base_pd, live_pd = get_methods_from_slice(lamar_HGE_df, RANSACParameters.methods_to_evaluate)
ransac_base_row, ransac_live_row, top_score_row = get_base_live_top_score_methods(base_pd, live_pd)
print_specific_row(ransac_base_row, ransac_base_row["Method Name"])
print_specific_row(ransac_live_row, ransac_live_row["Method Name"])
print_specific_row(top_score_row, top_score_row["Method Name"])
print_end_of_table()
all_top_score_method[top_score_row["Method Name"]].append(top_score_row['MAA'])

# Lamar - CAB
lamar_CAB_df = pd.read_csv(lamar_CAB_csv_file)
print_start_of_table(label= f"lamar_CAB_top_methods", caption="TODO: Add description")
base_pd, live_pd = get_methods_from_slice(lamar_CAB_df, RANSACParameters.methods_to_evaluate)
ransac_base_row, ransac_live_row, top_score_row = get_base_live_top_score_methods(base_pd, live_pd)
print_specific_row(ransac_base_row, ransac_base_row["Method Name"])
print_specific_row(ransac_live_row, ransac_live_row["Method Name"])
print_specific_row(top_score_row, top_score_row["Method Name"])
print_end_of_table()
all_top_score_method[top_score_row["Method Name"]].append(top_score_row['MAA'])

# Lamar - LIN
lamar_LIN_df = pd.read_csv(lamar_LIN_csv_file)
print_start_of_table(label= f"lamar_LIN_top_methods", caption="TODO: Add description")
ransac_base_row, ransac_live_row, top_score_row = get_base_live_top_score_methods(base_pd, live_pd)
print_specific_row(ransac_base_row, ransac_base_row["Method Name"])
print_specific_row(ransac_live_row, ransac_live_row["Method Name"])
print_specific_row(top_score_row, top_score_row["Method Name"])
print_end_of_table()
all_top_score_method[top_score_row["Method Name"]].append(top_score_row['MAA'])

print("Top performing for CMU, LaMAR, and Retail: (Do I need this ?)")
sorted_all_top_score_method = {}
for method_name, maa_list in all_top_score_method.items():
    sorted_all_top_score_method[method_name] = np.mean(maa_list)
sorted_all_top_score_method = {k: v for k, v in sorted(sorted_all_top_score_method.items(), key=lambda item: item[1], reverse=True)}
print(sorted_all_top_score_method)
print()

print(">>>>>>>>>>>>>>>>>>> Now printing base, live, and the top score method, one table for each array value below. Copy from below this line <<<<<<<<<<<<<<<<<<<<<<<<<<<")

# CMU
print_start_of_metric_table_aggregated_metrics(label="slices_metrics_for_base_live_top_score_method", caption="Selected Metrics")
for slice_name, methods_metrics in all_slice_metrics.items():
    print_aggregated_table_row(slice_name, methods_metrics)
print_end_of_table()

# Retail
base_pd, live_pd = get_methods_from_slice(retail_df, RANSACParameters.methods_to_evaluate)
ransac_base_row, ransac_live_row, top_score_row = get_base_live_top_score_methods(base_pd, live_pd)
metrics = {"base" : return_data_from_row(ransac_base_row, convert_to_cm=True), "live" : return_data_from_row(ransac_live_row, convert_to_cm=True), "top_score_method" : return_data_from_row(top_score_row, convert_to_cm=True)}
print_start_of_metric_table_aggregated_metrics(label="retail_shop_methods_base_live_top", caption="Retail Shop - top methods metrics")
print_aggregated_table_row(slice_name = None, methods_metrics = metrics)
print_end_of_table()

# Might need to modify the latex code a bit
# Lamar
base_pd, live_pd = get_methods_from_slice(lamar_HGE_df, RANSACParameters.methods_to_evaluate)
# Lamar - HGE
ransac_base_row, ransac_live_row, top_score_row = get_base_live_top_score_methods(base_pd, live_pd)
metrics = {"base" : return_data_from_row(ransac_base_row), "live" : return_data_from_row(ransac_live_row), "top_score_method" : return_data_from_row(top_score_row)}
print_start_of_metric_table_aggregated_metrics(label="lamar_HGE_methods_base_live_top", caption="HGE - top methods metrics")
print_aggregated_table_row(slice_name = None, methods_metrics = metrics)
# Lamar - CAB
base_pd, live_pd = get_methods_from_slice(lamar_CAB_df, RANSACParameters.methods_to_evaluate)
ransac_base_row, ransac_live_row, top_score_row = get_base_live_top_score_methods(base_pd, live_pd)
metrics = {"base" : return_data_from_row(ransac_base_row), "live" : return_data_from_row(ransac_live_row), "top_score_method" : return_data_from_row(top_score_row)}
print_aggregated_table_row(slice_name = None, methods_metrics = metrics)
# Lamar - LIN
base_pd, live_pd = get_methods_from_slice(lamar_LIN_df, RANSACParameters.methods_to_evaluate)
ransac_base_row, ransac_live_row, top_score_row = get_base_live_top_score_methods(base_pd, live_pd)
metrics = {"base" : return_data_from_row(ransac_base_row), "live" : return_data_from_row(ransac_live_row), "top_score_method" : return_data_from_row(top_score_row)}
print_aggregated_table_row(slice_name = None, methods_metrics = metrics)
print_end_of_table()

print("Done")