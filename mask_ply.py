import numpy as np
import open3d as o3d
import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=str, default="N3OpenArea")
args = parser.parse_args()

directory_pattern =  r'.*{}$'.format(args.n)

pcd = o3d.io.read_point_cloud('/local/home/yeltao/Desktop/3DHuman_gen/dataset/proxe/scenes/'+args.n+'.ply')
pcd_in_np = np.asarray(pcd.points)

num_files = 0

for dirpath, dirnames, filenames in os.walk('/local/home/yeltao/Downloads'):
    for dirname in dirnames:
        if re.match(directory_pattern, dirname):
            directory_path = os.path.join(dirpath, dirname, "pred_mask")
            for filename in os.listdir(directory_path):
                num_files += 1

def processMask(i):
    mask = []
    for dirpath, dirnames, filenames in os.walk('/local/home/yeltao/Downloads'):
        filename_pattern = r".*_{}\.txt$".format(i)
        for dirname in dirnames:
            if re.match(directory_pattern, dirname):
                directory_path = os.path.join(dirpath, dirname, "pred_mask")
                for filename in os.listdir(directory_path):
                    if re.match(filename_pattern, filename):
                        file_path = os.path.join(directory_path, filename)
                        with open(file_path, 'r') as f:
                            for line in f:
                                mask.append(int(line.strip('\n')))

    pcd_in_np_masked = [pcd_in_np[i] for i, v in enumerate(mask) if v == 1]
    pcd_masked = o3d.geometry.PointCloud()
    pcd_masked.points = o3d.utility.Vector3dVector(pcd_in_np_masked)
    o3d.io.write_point_cloud("scene_meshes/"+args.n+"_" + str(i) + ".ply", pcd_masked, write_ascii=True)
    
for i in range(num_files):
    processMask(i)

# for j in range(num_files):
#     pcd = o3d.io.read_point_cloud("scene_meshes/"+args.n+"_" + str(j) + ".ply")
#     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.1)
#     o3d.io.write_triangle_mesh("scene_meshes/"+args.n+"_"+str(j)+".obj", mesh, write_triangle_uvs=True)