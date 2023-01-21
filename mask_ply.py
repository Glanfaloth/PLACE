import numpy as np
import open3d as o3d

def processMask(i):
    pcd = o3d.io.read_point_cloud('/local/home/yeltao/Desktop/3DHuman_gen/dataset/proxe/scenes/N3OpenArea.ply')
    pcd_in_np = np.asarray(pcd.points)
    mask = []
    f = open('/local/home/yeltao/Downloads/20230118_173140_N3OpenArea/pred_mask/20230118_173140_N3OpenArea_' + str(i) + '.txt', "r")
    for line in f:
        mask.append(int(line.strip('\n')))

    pcd_in_np_masked = [pcd_in_np[i] for i, v in enumerate(mask) if v == 1]
    pcd_masked = o3d.geometry.PointCloud()
    pcd_masked.points = o3d.utility.Vector3dVector(pcd_in_np_masked)
    o3d.io.write_point_cloud("N3OpenArea_" + str(i) + ".ply", pcd_masked, write_ascii=True)
    pcd_new = o3d.io.read_point_cloud("N3OpenArea_" + str(i) + ".ply")
    o3d.visualization.draw_geometries([pcd_new])

for i in range(20):
    processMask(i)