import numpy as np
import time

from tqdm import tqdm

from RANSACParameters import RANSACParameters

# Example:  match_data = [[x, y, x, y, z , m.distance, n.distance], [i_m, i_n, s_m, s_n, v_m, v_n]] -> but flatten
# [[x (0), y (1), x (2), y (3), z (4), m.distance (5), n.distance (6)], [i_m (7), i_n (8), s_m (9), s_n (10), v_m (11), v_n (12)]]
# first value is of m (the closest match), second value is of n (second closest).
# i = per_image (prev. heatmap)
# s = per_session (prev. reliability)
# v = visibility

def get_sub_distribution(matches_for_image, index):
    vals = matches_for_image[:, index]
    sub_distribution = vals / np.sum(vals)
    sub_distribution = sub_distribution.reshape([sub_distribution.shape[0], 1])
    return sub_distribution

# lowes_distance_inverse = n.distance / m.distance  # inverse here as the higher the better for PROSAC
def lowes_distance_inverse(matches):
    return matches[:, 6] / matches[:, 5]

def per_image_score(matches):
    return matches[:, 7]

def per_session_score(matches):
    return matches[:, 9]

def visibility_score(matches):
    return matches[:, 11]

def per_session_score_ratio(matches):
    return np.nan_to_num(matches[:, 9] / matches[:, 10], nan = 0.0, neginf = 0.0, posinf = 0.0)

def per_image_score_ratio(matches):
    return matches[:, 7] / matches[:, 8]

def lowes_ratio_by_higher_per_session_score(matches):
    scores = []
    for match in matches:
        lowes_distance_inverse = match[6] / match[5]
        score_m = match[9]
        score_n = match[10]
        higher_score = score_m if score_m > score_n else score_n
        final_score = lowes_distance_inverse * higher_score
        scores.append(final_score)
    return np.array(scores)

def lowes_ratio_by_higher_per_image_score(matches):
    values = []
    for match in matches:
        lowes_distance_inverse = match[6] / match[5]
        val_m = match[7]
        val_n = match[8]
        higher_val = val_m if val_m > val_n else val_n
        final_score = lowes_distance_inverse * higher_val
        values.append(final_score)
    return np.array(values)

def lowes_ratio_per_session_score(matches):
    scores = []
    for match in matches:
        lowes_distance_inverse = match[6] / match[5]
        per_session_score_ratio = match[9] / match[10]
        final_score = lowes_distance_inverse * per_session_score_ratio
        scores.append(final_score)
    return np.array(scores)

def lowes_ratio_per_image_score(matches):
    scores = []
    for match in matches:
        lowes_distance_inverse = match[6] / match[5]
        per_image_score_ratio =  match[7] /  match[8]
        final_score = lowes_distance_inverse * per_image_score_ratio
        scores.append(final_score)
    return np.array(scores)

def higher_per_session_score(matches):
    scores = []
    for match in matches:
        score_m = match[9]
        score_n = match[10]
        higher_score = score_m if score_m > score_n else score_n
        scores.append(higher_score)
    return np.array(scores)

def higher_per_image_score(matches):
    values = []
    for match in matches:
        value_m = match[7]
        value_n = match[8]
        higher_value = value_m if value_m > value_n else value_n
        values.append(higher_value)
    return np.array(values)

def higher_visibility_score(matches):
    scores = []
    for match in matches:
        score_m = match[11]
        score_n = match[12]
        higher_score = score_m if score_m > score_n else score_n
        scores.append(higher_score)
    return np.array(scores)

functions = {RANSACParameters.lowes_distance_inverse_ratio_index : lowes_distance_inverse,
             RANSACParameters.per_image_score_index : per_image_score,
             RANSACParameters.per_session_score_index : per_session_score,
             RANSACParameters.per_session_score_ratio_index : per_session_score_ratio,
             RANSACParameters.lowes_ratio_per_session_score_index : lowes_ratio_per_session_score,
             RANSACParameters.lowes_ratio_per_image_score_index : lowes_ratio_per_image_score,
             RANSACParameters.higher_per_session_score_index : higher_per_session_score,
             RANSACParameters.per_image_score_ratio_index: per_image_score_ratio,
             RANSACParameters.higher_per_image_score_index: higher_per_image_score,
             RANSACParameters.higher_visibility_score_index: higher_visibility_score,
             RANSACParameters.lowes_ratio_by_higher_per_session_score_index: lowes_ratio_by_higher_per_session_score,
             RANSACParameters.lowes_ratio_by_higher_per_image_score_index: lowes_ratio_by_higher_per_image_score,
             RANSACParameters.visibility_score_index: visibility_score}

def sort_matches(matches, idx):
    score_list = functions[idx](matches)
    # sorted_indices
    sorted_indices = np.argsort(score_list)
    # in descending order ([::-1] makes it from ascending to descending )
    sorted_matches = matches[sorted_indices[::-1]]
    return sorted_matches

# 29/09/2022, This is used to return data per image! So I can examine later one by one! (as in Neural Filtering)
def run_comparison(func, matches, test_images, all_intrinsics, no_iterations, val_idx = None):

    #  this will hold inliers_no, outliers_no, iterations, time for each image
    images_data = {}

    for i in tqdm(range(len(test_images))):
        image = test_images[i]
        matches_for_image = matches[image]

        if (len(matches_for_image) < 10): #Not enough matches to get a reliable pose - mark as degenarate pose
            # est_pose, inliers_no, outliers_no, iterations, elapsed_time
            images_data[image] = [None, None, None, None, None]
            continue

        if (val_idx is not None):
            if (val_idx >= 0):
                matches_for_image = sort_matches(matches_for_image, val_idx)

            # These below are for RANSAC + dist versions
            if (val_idx == RANSACParameters.use_ransac_dist_per_image_score):
                sub_dist = get_sub_distribution(matches_for_image, 7)
                matches_for_image = np.hstack((matches_for_image, sub_dist))

            if (val_idx == RANSACParameters.use_ransac_dist_pre_session_score):
                sub_dist = get_sub_distribution(matches_for_image, 9)
                matches_for_image = np.hstack((matches_for_image, sub_dist))

            if (val_idx == RANSACParameters.use_ransac_dist_visibility_score):
                sub_dist = get_sub_distribution(matches_for_image, 11)
                matches_for_image = np.hstack((matches_for_image, sub_dist))

        intrinsics = all_intrinsics[image]
        start = time.time()
        best_model = func(matches_for_image, intrinsics, no_iterations) #this will return none if pose is not estimated

        if(best_model == None): #degenerate case
            images_data[image] = [None, None, None, None, None]
            continue

        end = time.time()
        elapsed_time = end - start

        est_pose = best_model['Rt']
        inliers_no = best_model['inliers_no']
        outliers_no = best_model['outliers_no']
        iterations = best_model['iterations']

        images_data[image] = [est_pose, inliers_no, outliers_no, iterations, elapsed_time]

    return images_data