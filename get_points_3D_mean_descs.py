# run this to get the avg of the 3D desc of a point same order as in points3D
# be careful that you can get the base model's avg descs or the live's model descs - depends on the points images ids

# the idea here is that a point is seen by the base model images and live model images
# obviously the live model images number > base model images number for a point
import sys
import numpy as np
from tqdm import tqdm
from database import COLMAPDatabase
from parameters import Parameters
from point3D_loader import read_points3d_default
from query_image import read_images_binary

SIZE = 129
def get_desc_avg(points3D, db):
    points_mean_descs_ids = np.empty([len(points3D.keys()), SIZE])

    point3D_vm_col_idx = 0
    for k,v in tqdm(points3D.items()):
        point_id = v.id
        points_image_ids = points3D[point_id].image_ids #COLMAP adds the image twice some times.
        points3D_descs = np.empty([len(points_image_ids), 128])
        # Loop through the points' image ids and check if it is seen by any image_ids
        # If it is seen then get the desc for each id.
        for k in range(len(points_image_ids)):
            id = points_image_ids[k]
            data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + str(id) + "'")
            data = COLMAPDatabase.blob_to_array(data.fetchone()[0], np.uint8)
            descs_rows = int(np.shape(data)[0] / 128)
            descs = data.reshape([descs_rows, 128]) #descs for the whole image
            keypoint_index = points3D[point_id].point2D_idxs[k]
            desc = descs[keypoint_index] #keypoints and descs are ordered the same (so I use the point2D_idxs to index descs )
            desc = desc.reshape(1, 128) #this is the desc of keypoint with index, keypoint_index, from image with id, id.
            points3D_descs[k] = desc

        # adding and calculating the mean here!
        mean = points3D_descs.mean(axis=0)
        points_mean_descs_ids[point3D_vm_col_idx] = np.append(mean, point_id).reshape(1,SIZE)
        point3D_vm_col_idx += 1
    return points_mean_descs_ids

# 08/12/2022 - base path should contain base, live, gt model
base_path = sys.argv[1] # example: "/home/alex/fullpipeline/colmap_data/CMU_data/slice2/" #trailing "/" or add "exmaps_data"
parameters = Parameters(base_path)

db_live = COLMAPDatabase.connect(parameters.live_db_path)
db_base = COLMAPDatabase.connect(parameters.base_db_path)

# TODO: 20/03/2012 - images Not needed ?
base_model_images = read_images_binary(parameters.base_model_images_path)
base_model_points3D = read_points3d_default(parameters.base_model_points3D_path)

live_model_images = read_images_binary(parameters.live_model_images_path)
live_model_points3D = read_points3d_default(parameters.live_model_points3D_path)

# You will notice that I am using live_model_points3D in both cases, fetching avg features for the base images and the live images.
# This is because the live_model_points3D points' images_ids hold also ids of the live and base model images, since the live model is just the
# base model with extra images localised in it. You can use the base model for the base images but you need to make sure that the base model is exactly the
# same as the live model, before you do. TODO: Maybe change to base to get it done with ?
# 08/12/2022 - it seems that you are not doing the above - you are using live and base seperately which makes more sense.

# 2 cases base and live images points3D descs
print("Getting base avg descs")
avgs_base = get_desc_avg(base_model_points3D, db_base)
np.save(parameters.avg_descs_base_path, avgs_base)

print("Getting live avg descs")
avgs_live = get_desc_avg(live_model_points3D, db_live)
np.save(parameters.avg_descs_live_path, avgs_live)
