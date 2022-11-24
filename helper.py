import glob
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm
from database import COLMAPDatabase
from undistort_img import undistort_cmu


def remove_folder_safe(folder_path):
    print(f"Deleting {folder_path}")
    if (os.path.exists(folder_path)):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)
    pass

def remove_folder(folder_path):
    print(f"Deleting {folder_path}")
    if (os.path.exists(folder_path)):
        shutil.rmtree(folder_path)
    pass

def remove_file_safe(path):
    if (os.path.isfile(path)):
        os.remove(path)

def empty_points_3D_txt_file(path):
    open(path, 'w').close()

def arrange_cameras_txt_file(db_path,path):
    db = COLMAPDatabase.connect(db_path)
    camera_model = db.execute("SELECT model FROM cameras WHERE camera_id = 1").fetchone()[0]
    assert camera_model == 1
    camera_params = db.execute("SELECT params FROM cameras WHERE camera_id = 1").fetchone()[0]
    camera_params = COLMAPDatabase.blob_to_array(camera_params, np.float64)
    fx = camera_params[0]
    fy = camera_params[1]
    cx = camera_params[2]
    cy = camera_params[3]
    width = db.execute("SELECT width FROM cameras WHERE camera_id = 1").fetchone()[0]
    height = db.execute("SELECT height FROM cameras WHERE camera_id = 1").fetchone()[0]
    # The value below is taken from the CMU documentation
    val = f'# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n# Number of cameras: 1\n' \
          f'1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}'
    with open(path, "w") as f:
        f.writelines(val)

def arrange_images_txt_file(db_path, images, path):
    db = COLMAPDatabase.connect(db_path)
    with open(path, "r") as f:
        lines = f.readlines()
    with open(path, "w") as f:
        for line in lines:
            cmu_image_name = line.split(" ")[-1].strip()
            if(cmu_image_name in images):
                if "#" in line:
                    f.write(line)
                if ".jpg" in line:
                    line = line.split(" ")
                    name = line[-1].strip()
                    id = db.execute("SELECT image_id FROM images WHERE name = " + "'" + str(name) + "'").fetchone()[0]
                    line[0] = str(id)
                    line = " ".join(line)
                    f.write(line)
                    f.write("\n")
                    id +=1

def arrange_sessions(source, dest):
    # From CMU README
    # Sunny + No Foliage (reference) | 4 Apr 2011
    # Sunny + Foliage | 1 Sep 2010
    # Sunny + Foliage | 15 Sep 2010
    # Cloudy + Foliage | 1 Oct 2010
    # Sunny + Foliage | 19 Oct 2010
    # Overcast + Mixed Foliage | 28 Oct 2010
    # Low Sun + Mixed Foliage | 3 Nov 2010
    # Low Sun + Mixed Foliage | 12 Nov 2010
    # Cloudy + Mixed Foliage | 22 Nov 2010
    # Low Sun + No Foliage + Snow | 21 Dec 2010
    # Low Sun + Foliage | 4 Mar 2011
    # Overcast + Foliage | 28 Jul 2011
    sessions_path = []
    for i in range(1,13,1):
        session_path = os.path.join(dest, f"session_{i}")
        remove_folder_safe(session_path)
        sessions_path.append(session_path)

    for file in tqdm(glob.glob(os.path.join(source,"*.jpg"))):
        cam_0 = "_c0_"
        if (cam_0 in file):
            timestamp = int(file.split(cam_0)[-1].split("us.jpg")[0])
            dt = datetime.fromtimestamp(timestamp / 1000000)
            day = dt.day
            month = dt.month
            head, tail = os.path.split(file)
            undistorted_img = undistort_cmu(cv2.imread(file))

            # one of the session folders will be the base so it will be always empty
            if (day == 4 and month == 4):
                cv2.imwrite(os.path.join(os.path.join(sessions_path[0], tail)), undistorted_img)
            if (day == 1 and month == 9):
                cv2.imwrite(os.path.join(os.path.join(sessions_path[1], tail)), undistorted_img)
            if (day == 15 and month == 9):
                cv2.imwrite(os.path.join(os.path.join(sessions_path[2], tail)), undistorted_img)
            if (day == 1 and month == 10):
                cv2.imwrite(os.path.join(os.path.join(sessions_path[3], tail)), undistorted_img)
            if (day == 19 and month == 10):
                cv2.imwrite(os.path.join(os.path.join(sessions_path[4], tail)), undistorted_img)
            if (day == 26 and month == 10):  # this should be 28/10 but I think they made a mistake it is 26/10
                cv2.imwrite(os.path.join(os.path.join(sessions_path[5], tail)), undistorted_img)
            if (day == 3 and month == 11):
                cv2.imwrite(os.path.join(os.path.join(sessions_path[6], tail)), undistorted_img)
            if (day == 12 and month == 11):
                cv2.imwrite(os.path.join(os.path.join(sessions_path[7], tail)), undistorted_img)
            if (day == 22 and month == 11):
                cv2.imwrite(os.path.join(os.path.join(sessions_path[8], tail)), undistorted_img)
            if (day == 21 and month == 12):
                cv2.imwrite(os.path.join(os.path.join(sessions_path[9], tail)), undistorted_img)
            if (day == 4 and month == 3):
                cv2.imwrite(os.path.join(os.path.join(sessions_path[10], tail)), undistorted_img)
            if (day == 28 and month == 7):
                cv2.imwrite(os.path.join(os.path.join(sessions_path[11], tail)), undistorted_img)

    return sessions_path

def create_query_image_names_txt(txt_path, images_path):
    with open(txt_path, 'w') as f:
        for image in glob.glob(os.path.join(images_path,'*/*.jpg'), recursive=True):
            image_loc = image.split(f"{images_path}/")[1]
            f.write(f"{image_loc}\n")
