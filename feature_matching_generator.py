# This is to match sfm images (already localised in COLMAP) descs against the
# trainDescriptors (which can from the base model or the complete model) as a benchmark and also exports the 2D-3D matches for ransac
# matching here is done using my own DM (direct matching) function.

# NOTE: One can argue why am I using the query images only (query_name.txt)? It makes sense more intuitively as
# I am localising the new (future sessions images) against a base model and a complete model. So the difference is in
# the model you are localising against.. But you could use all images. If you do then localising base images against the
# base model doesn't really makes sense, because at this point you are localising images the model has already seen but then again
# you can say the same thing for localising future images against the complete model
import cv2
import numpy as np
from point3D_loader import read_points3d_default, index_dict

# creates 2d-3d matches data for ransac comparison
def get_matches(good_matches_data, points3D_indexing, points3D, query_image_xy, points3D_scores):
    # same length
    # good_matches_data[0] - 2D point indices,
    # good_matches_data[1] - 3D point indices, - this is the index! you need the id to get xyz
    # good_matches_data[2] - lowe's distance inverse ratio
    # good_matches_data[3] - reliability scores ratio
    # good_matches_data[4] - reliability_scores = lowe's distance inverse * reliability scores ratio
    # good_matches_data[4] - score (the highest from the 2 nearest matches)
    data_size = 11
    matches = np.empty([0, data_size])
    for i in range(len(good_matches_data[1])):
        # get 2D point data
        xy_2D = query_image_xy[good_matches_data[0][i]] # remember these indices belong to query_image_xy

        # get 3D point data
        points3D_index = good_matches_data[1][i] # remember points3D_index is aligned with trainDescriptors
        points3D_id = points3D_indexing[points3D_index]
        xyz_3D = points3D[points3D_id].xyz

        # get lowe's inv ratio
        lowes_distance_inverse_ratio = good_matches_data[2][i]
        # reliability scores ratio
        reliability_score_ratio = good_matches_data[3][i]
        # reliability_scores
        reliability_score = good_matches_data[4][i]
        # the highest score of the 2 nearest points
        closest_neighbour_score = good_matches_data[5][i]

        # the score value (of the 3D point of the match, the closest match, m)
        points3D_score = points3D_scores[0, points3D_index]

        # values here are self explanatory..
        match = np.array([xy_2D[0], xy_2D[1], xyz_3D[0], xyz_3D[1], xyz_3D[2], points3D_index,
                          lowes_distance_inverse_ratio, points3D_score, reliability_score_ratio, reliability_score, closest_neighbour_score]).reshape([1, data_size])
        matches = np.r_[matches, match]
    return matches

# indexing is the same as points3D indexing for trainDescriptors
def feature_matcher_wrapper(points_scores, db_query, query_images, trainDescriptors, points3D, ratio_test_val, verbose = False):
    # create image_name <-> matches, dict - easier to work with
    matches = {}
    matches_sum = []
    points3D_indexing = index_dict(points3D)

    #  go through all the test images and match their descs to the 3d points avg descs
    for i in range(len(query_images)):
        query_image = query_images[i]
        if(verbose): print("Matching image " + str(i + 1) + "/" + str(len(query_images)) + ", " + query_image, end="\r")

        image_id = db_query.execute("SELECT image_id FROM images WHERE name = " + "'" + query_image + "'")
        image_id = str(image_id.fetchone()[0])

        # keypoints data
        query_image_keypoints_data = db_query.execute("SELECT data FROM keypoints WHERE image_id = " + "'" + image_id + "'")
        query_image_keypoints_data = query_image_keypoints_data.fetchone()[0]
        query_image_keypoints_data_cols = db_query.execute("SELECT cols FROM keypoints WHERE image_id = " + "'" + image_id + "'")
        query_image_keypoints_data_cols = int(query_image_keypoints_data_cols.fetchone()[0])
        query_image_keypoints_data = db_query.blob_to_array(query_image_keypoints_data, np.float32)
        query_image_keypoints_data_rows = int(np.shape(query_image_keypoints_data)[0] / query_image_keypoints_data_cols)
        query_image_keypoints_data = query_image_keypoints_data.reshape(query_image_keypoints_data_rows,query_image_keypoints_data_cols)
        query_image_keypoints_data_xy = query_image_keypoints_data[:, 0:2]

        # descriptors data
        query_image_descriptors_data = db_query.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + image_id + "'")
        query_image_descriptors_data = query_image_descriptors_data.fetchone()[0]
        query_image_descriptors_data = db_query.blob_to_array(query_image_descriptors_data, np.uint8)
        descs_rows = int(np.shape(query_image_descriptors_data)[0] / 128)
        query_image_descriptors_data = query_image_descriptors_data.reshape([descs_rows, 128])

        # once you have the test images descs now do feature matching here! - Matching on trainDescriptors (remember these are the means of the 3D points)
        queryDescriptors = query_image_descriptors_data.astype(np.float32)

        # actual matching here!
        # NOTE: 09/06/2020 - match() has been changed to return lowes_distances in REVERSE! (https://willguimont.github.io/cs/2019/12/26/prosac-algorithm.html)
        # good_matches = matcher.match(queryDescriptors, trainDescriptors)

        # NOTE: 03/07/2020 - using matching method from OPENCV
        bf = cv2.BFMatcher()
        temp_matches = bf.knnMatch(queryDescriptors, trainDescriptors, k=2)

        # output: idx1, idx2, lowes_distance (vectors of corresponding indexes in
        # queryDescriptors and trainDescriptors, and lowes_distances inverse respectively)
        idx1, idx2, lowes_distances, scores_ratios, custom_scores, closest_neighbour_scores  = [], [], [], [], [], []
        # m the closest, n is the second closest
        for m, n in temp_matches: # TODO: maybe consider what you have at this point? and add it to the if condition ?
            # trainIdx is from 0 to no of points 3D (since each point 3D has a desc), so you can use it as an index here
            score_m = points_scores[0, m.trainIdx]
            score_n = points_scores[0, n.trainIdx]
            if (m.distance < ratio_test_val * n.distance): #and (score_m > score_n):
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                lowes_distance_inverse = n.distance / m.distance #inverse here as the higher the better for PROSAC
                lowes_distances.append(lowes_distance_inverse)
                score_ratio = score_m / score_n # the higher the better (first match is more "static" than the second, ratio)
                scores_ratios.append(score_ratio)
                custom_score = lowes_distance_inverse * score_ratio # self-explanatory
                custom_scores.append(custom_score)
                closest_neighbour_score = score_m if score_m > score_n else score_n
                closest_neighbour_scores.append(closest_neighbour_score)
        # at this point you store 1, 2D - 3D match.
        good_matches = [idx1, idx2, lowes_distances, scores_ratios, custom_scores, closest_neighbour_scores]
        # queryDescriptors and query_image_keypoints_data_xy = same order
        # points3D order and trainDescriptors_* = same order
        # returns extra data for each match
        matches[query_image] = get_matches(good_matches, points3D_indexing, points3D, query_image_keypoints_data_xy, points_scores)
        matches_sum.append(len(good_matches[0]))

    if(verbose):
        print()
        total_all_images = np.sum(matches_sum)
        print("Total matches: " + str(total_all_images) + ", no of images " + str(len(query_images)))
        matches_all_avg = total_all_images / len(matches_sum)
        print("Average matches per image: " + str(matches_all_avg) + ", no of images " + str(len(query_images)))

    return matches

