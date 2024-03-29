class RANSACParameters(object):
    # PROSAC sorting values for matches indices
    lowes_distance_inverse_ratio_index = 0
    higher_visibility_score_index = 1
    per_image_score_index = 2
    per_session_score_ratio_index = 3
    lowes_ratio_per_session_score_index = 4
    higher_per_session_score_index = 5
    per_session_score_index = 6
    per_image_score_ratio_index = 7
    higher_per_image_score_index = 8
    lowes_ratio_per_image_score_index = 9
    lowes_ratio_by_higher_per_session_score_index = 10
    lowes_ratio_by_higher_per_image_score_index = 11
    visibility_score_index = 12

    # values used to sort matches, check sort_matches() function in ransac_comparison.py
    # 06/02/2023: removed some methods
    # prosac_value_titles = {lowes_distance_inverse_ratio_index: "prosac_lowes_distance_inverse",
    #                        per_image_score_index: "prosac_per_image_score",
    #                        per_session_score_index: "prosac_per_session_score",
    #                        per_session_score_ratio_index: "prosac_per_session_score_ratio",
    #                        lowes_ratio_per_session_score_index: "prosac_lowes_ratio_per_session_score",
    #                        lowes_ratio_per_image_score_index: "prosac_lowes_ratio_per_image_score",
    #                        higher_per_session_score_index: "prosac_higher_per_session_score",
    #                        per_image_score_ratio_index: "prosac_per_image_score_ratio",
    #                        higher_per_image_score_index: "prosac_higher_per_image_score",
    #                        higher_visibility_score_index: "prosac_higher_visibility_score",
    #                        lowes_ratio_by_higher_per_session_score_index: "prosac_lowes_ratio_by_higher_per_session_score",
    #                        lowes_ratio_by_higher_per_image_score_index: "prosac_lowes_ratio_by_higher_per_image_score",
    #                        visibility_score_index: "prosac_visibility_score"}
    # 06/02/2023: removed some methods (original above)
    prosac_value_titles = {lowes_distance_inverse_ratio_index: "prosac_lowes_distance_inverse",
                           per_image_score_index: "prosac_per_image_score",
                           per_session_score_index: "prosac_per_session_score",
                           per_session_score_ratio_index: "prosac_per_session_score_ratio",
                           per_image_score_ratio_index: "prosac_per_image_score_ratio",
                           visibility_score_index: "prosac_visibility_score"}

    use_ransac_dist_per_image_score = -1
    use_ransac_dist_per_session_score = -2
    use_ransac_dist_visibility_score = -3

    ransac_base = "ransac_base"
    prosac_base = "prosac_base"
    ransac_live = "ransac_live"
    ransac_dist_per_image_score = "ransac_dist_per_image_score"
    ransac_dist_per_session_score = "ransac_dist_per_session_score"
    ransac_dist_visibility_score = "ransac_dist_visibility_score"

    ransac_prosac_error_threshold = 5.0  # 8.0 default, 5 from the Benchamrking 6DoF long term localization

    # 31/01/2023 - adding specific methods to examine (I wanted to avoid the ratio methods I came up with - can't justify them)
    # first set = ["prosac_lowes_distance_inverse", "prosac_per_image_score", "prosac_per_session_score", "prosac_visibility_score", "ransac_base",
    #                            "prosac_base", "ransac_live", "ransac_dist_per_image_score", "ransac_dist_per_session_score", "ransac_dist_visibility_score",
    #                            "prosac_per_image_score_ratio", "prosac_per_session_score_ratio"]
    methods_to_evaluate = ["prosac_lowes_distance_inverse", "prosac_per_image_score", "prosac_per_session_score", "prosac_visibility_score", "ransac_base",
                           "prosac_base", "ransac_live", "ransac_dist_per_image_score", "ransac_dist_per_session_score", "ransac_dist_visibility_score",
                           "prosac_per_image_score_ratio", "prosac_per_session_score_ratio"]