import numpy as np
import pandas as pd

latex_dict_for_methods = { "prosac_lowes_distance_inverse" : "PROSAC \\cite{1467271}  $(d_{2}/d_{1})$",
                           "prosac_per_image_score" : "PROSAC with $\\stabi_1$",
                           "prosac_per_session_score" : "PROSAC with $\\stabs_1$",
                           "prosac_per_session_score_ratio" : "PROSAC with $(\\stabs_1 / \\stabs_2) $",
                           "prosac_lowes_ratio_per_session_score" : "PROSAC with $\\stabsLR$",
                           "prosac_lowes_ratio_per_image_score" : "PROSAC with $\\stabiLR$",
                           "prosac_higher_per_session_score" : "PROSAC with $\\max(\\stabs_1, \\stabs_2)$",
                           "prosac_per_image_score_ratio" : "PROSAC with $(\\stabi_1 / \\stabi_2) $",
                           "prosac_higher_per_image_score" : "PROSAC with $\\max(\\stabi_1, \\stabi_2)$",
                           "prosac_higher_visibility_score" : "PROSAC with $\\max(v_1 , v_2)$",
                           "prosac_lowes_ratio_by_higher_per_session_score" : "PROSAC with $\\max(\\stabs_1, \\stabs_2)LR$",
                           "prosac_lowes_ratio_by_higher_per_image_score" : "PROSAC with $\\max(\\stabi_1, \\stabi_2)LR$",
                           "prosac_visibility_score" : "PROSAC with $v_1$",
                           "ransac_base" : "RANSAC \\cite{10.1145/358669.358692}",
                           "prosac_base" : "PROSAC \\cite{1467271}  $(d_{2}/d_{1})$",
                           "ransac_live" : "RANSAC \\cite{10.1145/358669.358692}",
                           "ransac_dist_per_image_score" : "RANSAC with $\\stabi_1$",
                           "ransac_dist_per_session_score" : "RANSAC with $\\stabs_1$",
                           "ransac_dist_visibility_score" : "RANSAC with $v_1$"
                           }

def print_start_of_metric_table_aggregated_metrics(label, caption):
    print("\\begin{table} % [h]")
    print("\\caption{\\label{tab:"+label+"}%")
    print(caption)
    print("}\\vspace{0.5em}")
    print("\\centerline{")
    print("\\begin{tabular}{l@{\\hspace{5mm}}rr@{\\hspace{5mm}}rr@{\\hspace{4mm}}rr@{\\hspace{4mm}}rr}")
    print("\\toprule")
    print("Slice \\# & \\multicolumn{2}{l}{Map Matches} & \\multicolumn{2}{c}{RANSAC Base} & \\multicolumn{2}{c}{RANSAC Live} & \\multicolumn{2}{c}{Top Scored Method} \\\\")
    print("\\ & Base & Live & In.(\\%) & mAA(\\%) & In.(\\%) & mAA(\\%) & In.(\\%) & mAA(\\%) \\\\ \midrule")

def print_start_of_table(label, caption):
    print("\\begin{table} % [h]")
    print("\\caption{\\label{tab:"+label+"}%")
    print(caption)
    print("}\\vspace{0.5em}")
    print("\\centerline{")
    print("\\begin{tabular}{l@{\\hspace{2mm}}rrrrrrrr}")
    print("\\toprule")
    print("\\multicolumn{1}{c}{} & Total M. & Inl. (\%) & Outl. (\%) & Iters & Time (ms) & Er.[m] & Er.[°] & mAA(\\%) \\\\ \\midrule")

def print_end_of_table():
    print("\\bottomrule")
    print("\\end{tabular}")
    print("}") #for print("\\centerline{")
    print("\\end{table}")
    print()
    pass

def clear_page():
    print("\\clearpage")
    print("")

def print_aggregated_table_row(slice_name = None, methods_metrics= None):
    base_method = methods_metrics["base"]
    live_method = methods_metrics["live"]
    top_score_method = methods_metrics["top_score_method"]
    live_mAA = live_method[-1]
    top_score_mAA = top_score_method[-1]
    live_mAA_text = f"{live_method[-1]}"
    top_score_mAA_text = f"{top_score_method[-1]}"
    if(live_mAA > top_score_mAA):
        live_mAA_text = f"\\textbf{{{live_mAA}}}"
    if (live_mAA < top_score_mAA):
        top_score_mAA_text = f"\\textbf{{{top_score_mAA}}}"
    if (live_mAA == top_score_mAA):
        live_mAA_text = f"\\textbf{{{live_mAA}}}"
        top_score_mAA_text = f"\\textbf{{{top_score_mAA}}}"
    if(slice_name == None):
        slice_no = "1"
    else:
        slice_no = slice_name.split("e")[1]
    print(f"{slice_no} & {base_method[0]} & \\textbf{{{live_method[0]}}} & {base_method[1]} & {base_method[-1]} & {live_method[1]} & {live_mAA_text} & {top_score_method[1]} & {top_score_mAA_text} \\\\")

# This print rows and return the live selected methods - ranked by mAA
def print_all_rows(base_pd, live_pd, title):
    convert_to_cm = (title == "Retail Shop")
    print(f"{title}                          &    &     &     &      &      &      &      &     \\\\")
    print("\\textbf{Base Map}                &    &     &     &      &      &      &      &     \\\\")

    for _, base_row in base_pd.iterrows():
        method_name = base_row['Method Name']
        total_matches, inliers_perc, outliers_perc, iterations, total_time, translation_error, rotation_error, mAA = return_data_from_row(base_row, convert_to_cm=convert_to_cm)
        if (method_name == "ransac_base"):
            print(
                f"{latex_dict_for_methods[method_name]}               & {total_matches} & {inliers_perc} & {outliers_perc} & {iterations} "
                f"& {total_time} & {translation_error} & {rotation_error} & {mAA} \\\\")
        if (method_name == "prosac_base"):
            print(
                f"{latex_dict_for_methods[method_name]}               & {total_matches} & {inliers_perc} & {outliers_perc} & {iterations} "
                f"& {total_time} & {translation_error} & {rotation_error} & {mAA} \\\\")
    print("\\midrule")
    print("\\textbf{Live Map}                &    &     &     &      &      &      &      &     \\\\")
    idx = 0 #just to track the first one
    for _, live_row in live_pd.iterrows():
        idx += 1
        method_name = live_row['Method Name']
        total_matches, inliers_perc, outliers_perc, iterations, total_time, translation_error, rotation_error, mAA = return_data_from_row(live_row, convert_to_cm=convert_to_cm)
        if(idx == 1):
            mAA_latex = f"\\textbf{{{mAA}}}" #first one is bold
        else:
            mAA_latex = f"{mAA}"
        print(f"{latex_dict_for_methods[method_name]}                & {total_matches} & {inliers_perc} & {outliers_perc} & {iterations} "
              f"& {total_time} & {translation_error} & {rotation_error} & {mAA_latex}  \\\\")

def get_base_live_top_score_methods(base_pd, live_pd):
    # base
    method_name = "ransac_base"
    ransac_base_row = base_pd[base_pd["Method Name"] == method_name].iloc[0]
    # live
    method_name = "ransac_live"
    ransac_live_row = live_pd[live_pd["Method Name"] == method_name].iloc[0]
    # top method that uses a score
    method_name = live_pd["Method Name"][0] #already sorted by mAA
    top_score_row = live_pd[live_pd["Method Name"] == method_name].iloc[0]
    return ransac_base_row, ransac_live_row, top_score_row

def print_specific_row(row, method_name):
    total_matches, inliers_perc, outliers_perc, iterations, total_time, translation_error, rotation_error, mAA = return_data_from_row(row)
    print(f"{latex_dict_for_methods[method_name]} & {total_matches} & {inliers_perc} & {outliers_perc} & {iterations} & {total_time} & {translation_error} & {rotation_error} & {mAA} \\\\")

def append_to_metric_tables(metric_tables, total_matches, inliers_perc, outliers_perc, iterations, total_time, translation_error, rotation_error, mAA):
    metric_tables["Total Matches"].append(total_matches)
    metric_tables["Inliers (%)"].append(inliers_perc)
    metric_tables["Outliers (%)"].append(outliers_perc)
    metric_tables["Iterations"].append(iterations)
    metric_tables["Total Time (ms)"].append(total_time)
    metric_tables["Trans Error (m)"].append(translation_error)
    metric_tables["Rotation Error (°)"].append(rotation_error)
    metric_tables["mAA"].append(mAA)
    pass #no need to return anything, metric_tables is passed by reference

# works for all
def get_latex_row_string_top_methods(metric, base, live, top_score_method): #deal with cases when you have equals in Latex
    metric_latex = metric.replace("%", "\%")

    if metric in ["Outliers (%)", "Iterations", "Total Time (ms)", "Trans Error (m)", "Rotation Error (°)"]:  # we want the min of this, not max
        idx = np.argmin([base, live, top_score_method])
    else:
        idx = np.argmax([base, live, top_score_method])

    base_latex = f"\\textbf{{{base}}}" if idx == 0 else base
    live_latex = f"\\textbf{{{live}}}" if idx == 1 else live
    top_score_method_latex = f"\\textbf{{{top_score_method}}}" if idx == 2 else top_score_method

    return metric_latex , base_latex , live_latex , top_score_method_latex

def return_data_from_row(row, convert_to_cm=False):
    total_matches = int(np.round(row["Total Matches"]))
    inliers_perc = np.round(row["Inliers (%)"])
    outliers_perc = np.round(row["Outliers (%)"])
    iterations = int(row["Iterations"])
    total_time = np.round(row["Total Time (s)"] * 1000, decimals=3) # to ms (but keep the (s) to index the dict)
    if(convert_to_cm):
        translation_error = np.round(row["Trans Error (m)"]*100, decimals=3) #to cm
    else:
        translation_error = np.round(row["Trans Error (m)"], decimals=3) #to m
    rotation_error = np.round(row["Rotation Error (d)"], decimals=3)
    mAA = np.round(row["MAA"] * 100 , decimals=3) # to %
    return f"{(total_matches):.0f}", f"{(inliers_perc):.0f}", f"{(outliers_perc):.0f}", \
        f"{(iterations):.0f}", f"{(total_time):.2f}", f"{(translation_error):.2f}", \
        f"{(rotation_error):.2f}", f"{(mAA):.2f}"

def get_methods_from_slice_cmu(sub_frame, methods_to_evaluate): #already sorts my MAA
    base_rows = []
    live_rows = []
    for index, row in sub_frame.reset_index().iterrows():

        method_name = row["Method Name"]

        if ("base" in method_name):
            base_rows.append(row)
        else:
            if (method_name in methods_to_evaluate):
                live_rows.append(row)

    base_pd = pd.DataFrame.from_records(base_rows)
    live_pd = pd.DataFrame.from_records(live_rows)

    base_pd = base_pd.sort_values("MAA" , ascending=False)
    live_pd = live_pd.sort_values("MAA" , ascending=False)

    return base_pd, live_pd

def get_methods_from_slice(df, methods_to_evaluate): #for retail and HGE,  #already sorts my MAA
    base_rows = []
    live_rows = []
    for index, row in df.iterrows():
        method_name = row[0]
        if ("base" in method_name):
            base_rows.append(row)
        else:
            if (method_name in methods_to_evaluate):
                live_rows.append(row)

    base_pd = pd.DataFrame.from_records(base_rows)
    live_pd = pd.DataFrame.from_records(live_rows)

    base_pd = base_pd.sort_values("MAA", ascending=False)
    live_pd = live_pd.sort_values("MAA", ascending=False)

    return base_pd, live_pd