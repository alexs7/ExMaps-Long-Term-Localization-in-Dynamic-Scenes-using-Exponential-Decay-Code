# NOTE: 09/12/2022 This file was added to apply the exponential decay, get 3D points descs average, and run the main.py
# run this on the CYENS machine
import subprocess

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