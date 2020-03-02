from point3D_loader import get_points3D
from query_image import get_query_image_id_new_model
from evaluator import get_ARCore_pose_query_image
from query_image import get_query_image_global_pose_new_model
import numpy as np
import os
import sys

if(len(sys.argv) == 2 ):
    scale = float(sys.argv[1])
else:
    scale = 1

print("Scale: " + str(scale))

image_id_start = get_query_image_id_new_model("query.jpg")
points3D = get_points3D(image_id_start)
print("Number of COLMAP 3D Points: " + str(len(points3D)))

colmap_pose = get_query_image_global_pose_new_model("query.jpg")

print("COLMAP Pose: ")
print(colmap_pose)

colmap_to_arcore_matrix = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0]])
colmap_to_arcore_matrix = scale * colmap_to_arcore_matrix
colmap_to_arcore_matrix = np.r_[colmap_to_arcore_matrix, [np.array([0, 0, 0, 1])]]

print("colmap_to_arcore_matrix: ")
print(colmap_to_arcore_matrix)

rotZ = np.array([[0, 1, 0, 0],
                 [-1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])

arcore_pose = get_ARCore_pose_query_image()
print("arcore_pose: ")
print(arcore_pose)

arcore_pose_inverse = np.linalg.inv(arcore_pose)
print("arcore_pose_inverse: ")
print(arcore_pose_inverse)

#from_colmap_world_to_colmap_camera
points3D = colmap_pose.dot(np.transpose(points3D))
#from_colmap_camera_to_arcore_camera
points3D = colmap_to_arcore_matrix.dot(rotZ.dot(points3D))
#from_arcore_camera_to_arcore_world
points3D = arcore_pose_inverse.dot(points3D)
points3D = np.transpose(points3D)

os.system("rm /Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/points3D_AR.txt")
np.savetxt('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/points3D_AR.txt', points3D)