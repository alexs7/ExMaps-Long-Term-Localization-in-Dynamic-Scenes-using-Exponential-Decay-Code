# applies the exponential decay on images - not 3D points as it was before!
# 16/01/2023 - This is the version with the memory improvements
import collections
import sys
import numpy as np
from tqdm import tqdm
from parameters import Parameters
from point3D_loader import read_points3d_default
from query_image import read_images_binary, get_col, get_points3D_ids_row, get_images_map_id_idx

# 0 MUST be base session then 1 next session etc etc...
def get_db_sessions(no_images_per_session):
    sessions = {}
    images_traversed = 0
    for i in range(len(no_images_per_session)):
        no_images = no_images_per_session[i]
        image_ids = []
        start = images_traversed
        end = start + no_images
        for k in range(start,end):
            id = k + 1 # to match db ids
            image_ids.append(id)
            images_traversed += 1
        sessions[i] = image_ids
    return sessions

def create_vm(parameters):
    # by "live model" I mean all the frames from future sessions localised in the base model, including images from base model
    live_model_all_images = read_images_binary(parameters.live_model_images_path)

    # Sort the images by id in ascending order - represents the order they were localised in the live model
    # In other words follow the same order as in the database
    # It might not make a difference for the per image score, but it might do for the per session score
    live_model_all_images = collections.OrderedDict(sorted(live_model_all_images.items()))

    live_model_points3D = read_points3d_default(parameters.live_model_points3D_path)  # live model's 3D points (same length as base (but different order from base!) as we do not add points when localising new points, but different image_ds for each point)
    base_model_points3D = read_points3d_default(parameters.base_model_points3D_path)  # live model's 3D points (same length as base (but different order from base!) as we do not add points when localising new points, but different image_ds for each point)

    sessions_numbers = np.loadtxt(parameters.no_images_per_session_path).astype(int) #this was generated by counting the number of image files (.jpg), should be the same in db
    sessions_from_db = get_db_sessions(sessions_numbers)  # session_index -> [images_ids]

    print("First Loop..")
    # third loop is to build the binary VM matrix

    live_images_no = len(live_model_all_images.keys())
    images_map_id_idx = get_images_map_id_idx(live_model_all_images) #only localised images

    # per image decay weights (and visibility scores)
    total_sessions = len(sessions_from_db.keys())
    t1_2_custom = int(live_images_no / total_sessions)
    t_index = np.arange(live_images_no - 1, -1, -1)
    t_index = 0.5 ** ((t_index + 1) / t1_2_custom)  # add plus one here because points in the database are already decayed

    per_image_decay_scores = np.zeros([1, len(live_model_points3D)])
    visibility_scores = np.zeros([1, len(live_model_points3D)])
    # save the 3D points' index and column that states from which images it has been seen
    # as a ROW in a database. Then I can use the 3D points' index to query the database and fetch that row
    # it would be easier this way to apply the decay per column and get the sum of all columns (scores)
    point3D_vm_col_idx = 0
    for point_id, point in tqdm(live_model_points3D.items()):
        assert point_id == point.id
        vm_col = get_col(point, images_map_id_idx, live_model_all_images, 1)
        visibility_scores[0, point3D_vm_col_idx] = vm_col.sum()  # just the visibility scores
        # per image decay is here
        final_col = vm_col.reshape(vm_col.shape[0],) * t_index #apply weight on column
        per_image_decay_scores[0, point3D_vm_col_idx] = final_col.sum() #same sum, for this point3D
        point3D_vm_col_idx += 1

    #per session decay weights
    per_session_decay_scores = np.zeros([1, len(live_model_points3D)])
    N0 = 1  #default value, if a point is seen from an image
    t1_2 = 1  # 1 day
    weight_col = np.zeros([len(live_model_all_images),])
    weight_col_idx = 0
    for sessions_no, session_image_ids in tqdm(sessions_from_db.items()): #ordered (base, s1, s2 ... end)
        # session_index -> [images_ids] here we have sessions_no -> image_ids (same)
        # NOTE: the image_ids are from the db (more than the localised).
        t = len(sessions_from_db) - (sessions_no + 1) + 1 #since zero-based (14/07/2020, need to add one so it starts from the last number and goes down..)
        Nt = N0 * (0.5) ** (t / t1_2)
        for image_id in session_image_ids: #all image_ids in a session starting from the first to last (most recent) session, so Nt will only be applied to those images
            if image_id in images_map_id_idx: #if the image (id) is localised, then save the weight, for this image
                assert weight_col_idx == images_map_id_idx[image_id]
                weight_col[weight_col_idx] = Nt
                weight_col_idx += 1

    # apply the session decay
    point3D_vm_col_idx = 0
    for point_id, point in tqdm(live_model_points3D.items()):
        assert point_id == point.id
        vm_col = get_col(point, images_map_id_idx, live_model_all_images, 1)
        final_col = vm_col.reshape(vm_col.shape[0], ) * weight_col  # apply weight on column
        per_session_decay_scores[0, point3D_vm_col_idx] = final_col.sum()  # same sum, for this point3D
        point3D_vm_col_idx += 1

    print("Saving files...")
    np.save(parameters.per_image_decay_scores_path, per_image_decay_scores)
    np.save(parameters.per_session_decay_scores_path, per_session_decay_scores)
    np.save(parameters.binary_visibility_scores_path, visibility_scores)

    # 29/12/2022 - save the points ids too to double check in feature matcher that tha scores are assigned correctly
    np.save(parameters.live_points_3D_ids_file_path, get_points3D_ids_row(live_model_points3D))
    np.save(parameters.base_points_3D_ids_file_path, get_points3D_ids_row(base_model_points3D))

# 07/12/2022: base path should contain base, live, gt model
base_path = sys.argv[1] # example: "/home/alex/fullpipeline/colmap_data/CMU_data/slice2/" or add "exmaps_data". This value might come from another caller script
parameters = Parameters(base_path)
create_vm(parameters)
