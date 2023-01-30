from itertools import chain
import cv2
import numpy as np
from tqdm import tqdm
from query_image import get_image_id, get_keypoints_xy, get_queryDescriptors

def feature_matcher_wrapper(db, query_images, trainDescriptors_and_points3D_ids, points3D_xyz_ids, ratio_test_val, points_3D_ids, gt_images_only_to_use, points3D_bin, points_scores_array=None):
    # create image_name <-> matches, dict - easier to work with
    matches = {}
    # This dict will hold , image_name -> [number of 3D points its keypoints match to]
    keypoints_points3D = {}
    trainDescriptors = trainDescriptors_and_points3D_ids[:,0:128]
    points_3D_ids_from_train = trainDescriptors_and_points3D_ids[:,128]

    for i in range(len(points_3D_ids_from_train)): #sanity check no.1 that all is in order
        assert points_3D_ids_from_train[i] == points_3D_ids[0,i]

    #  go through all the test images and match their descs to the 3d points avg descs
    for i in tqdm(range(len(query_images))):
        query_image = query_images[i]

        image_id = get_image_id(db,query_image)
        # keypoints data - keypoints_xy and queryDescriptors have same order
        keypoints_xy = get_keypoints_xy(db, image_id)
        queryDescriptors = get_queryDescriptors(db, image_id)

        matcher = cv2.BFMatcher()  # cv2.FlannBasedMatcher(Parameters.index_params, Parameters.search_params) # or cv.BFMatcher()

        # Matching on trainDescriptors (remember these are the means of the 3D points)
        temp_matches = matcher.knnMatch(queryDescriptors, trainDescriptors, k=2)

        assert len(temp_matches) == keypoints_xy.shape[0] == queryDescriptors.shape[0] #another fucking sanity check

        # output: idx1, idx2, lowes_distance (vectors of corresponding indexes in
        # m the closest, n is the second closest
        good_matches = []
        point_3D_seen_counter = 0
        for m, n in temp_matches: # TODO: maybe consider what you have at this point? and add it to the if condition ?
            assert(m.distance <=  n.distance)
            # trainIdx is from 0 to no of points 3D (since each point 3D has a desc), so you can use it as an index here
            if (m.distance < ratio_test_val * n.distance): #and (score_m > score_n):
                if(m.queryIdx >= keypoints_xy.shape[0]): #keypoints_xy.shape[0] always same as queryDescriptors.shape[0]
                    raise Exception("m.queryIdx error!")
                if (m.trainIdx >= points3D_xyz_ids.shape[0]):
                    raise Exception("m.trainIdx error!")
                # idx1.append(m.queryIdx)
                # idx2.append(m.trainIdx)
                scores = []
                xy2D = keypoints_xy[m.queryIdx, :].tolist()
                xyz3D = points3D_xyz_ids[m.trainIdx, 0:3].tolist() #x,y,z, id
                point_id = points3D_xyz_ids[m.trainIdx, 3] #3D point id

                # sanity check no.2
                assert points_3D_ids_from_train[m.trainIdx] == points_3D_ids[0, m.trainIdx] == point_id

                if (points_scores_array is not None):
                    for points_scores in points_scores_array:
                        scores.append(points_scores[0, m.trainIdx]) # m , closest first neighbour's score
                        scores.append(points_scores[0, n.trainIdx]) # n , closest second neighbour's score

                if( point_id in gt_images_only_to_use[int(image_id)].point3D_ids):
                    point_3D_seen_counter += 1

                match_data = [xy2D, xyz3D, [m.distance, n.distance], scores]

                match_data = list(chain(*match_data))
                good_matches.append(match_data)

        keypoints_points3D[query_image] = point_3D_seen_counter
        matches[query_image] = np.array(good_matches)

    return matches, keypoints_points3D
