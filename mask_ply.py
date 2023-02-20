import numpy as np
import open3d as o3d
import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str, default="N3OpenArea")
args = parser.parse_args()

directory_pattern =  r'.*{}$'.format(args.scene)
directory = "scene_meshes/{}".format(args.scene)

if not os.path.exists(directory):
    os.mkdir(directory)

pcd = o3d.io.read_point_cloud('/local/home/yeltao/Desktop/3DHuman_gen/dataset/proxe/scenes/'+args.scene+'.ply')
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
    o3d.io.write_point_cloud(directory + "/"+args.scene+"_" + str(i) + ".ply", pcd_masked, write_ascii=True)
    
    
for i in range(num_files):
    processMask(i)

vis = []
for j in range(num_files):
    pcd_mesh = o3d.io.read_point_cloud(directory + "/" +args.scene+"_" + str(j) + ".ply")
    
    # Estimate normal vectors
    pcd_mesh.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Create mesh from point cloud
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_mesh,
        depth=10,
        scale=1.1,
        linear_fit=False)[0]

    o3d.io.write_triangle_mesh(directory + "/"+args.scene+"_"+str(j)+".obj", mesh)
    mesh_vis = o3d.io.read_triangle_mesh(directory + "/"+args.scene+"_"+str(j)+".obj")
    vis.append(mesh_vis)
o3d.visualization.draw_geometries(vis)