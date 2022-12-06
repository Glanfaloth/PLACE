import warnings
warnings.simplefilter("ignore", UserWarning)

import open3d as o3d
from models.cvae import *
from preprocess.preprocess_optimize import *
from preprocess.bps_encoding import *
from utils import *
from utils_read_data import *

prox_dataset_path = '/local/home/yeltao/Desktop/3DHuman_gen/dataset/proxe'
scene_name = 'N3OpenArea'

# read scen mesh/sdf
scene_mesh, cur_scene_verts, s_grid_min_batch, s_grid_max_batch, s_sdf_batch = read_mesh_sdf(prox_dataset_path,
                                                                                             'prox',
                                                                                             scene_name)
rot_angle_1, scene_min_x, scene_max_x, scene_min_y, scene_max_y = define_scene_boundary('prox', scene_name)

# draw box
mesh_box_1 = o3d.geometry.TriangleMesh.create_box(width=abs(scene_min_x)+abs(scene_max_x), height=0.1, depth=2.5)
mesh_box_2 = o3d.geometry.TriangleMesh.create_box(width=abs(scene_min_x)+abs(scene_max_x), height=0.1, depth=2.5)
mesh_box_3 = o3d.geometry.TriangleMesh.create_box(width=0.1, height=abs(scene_min_y)+abs(scene_max_y), depth=2.5)
mesh_box_4 = o3d.geometry.TriangleMesh.create_box(width=0.1, height=abs(scene_min_y)+abs(scene_max_y), depth=2.5)
mesh_box_1.paint_uniform_color([0.9, 0.1, 0.1]) # red
mesh_box_2.paint_uniform_color([0.1, 0.9, 0.1]) # green
mesh_box_3.paint_uniform_color([0.1, 0.1, 0.9]) # blue
mesh_box_4.paint_uniform_color([0.9, 0.9, 0.1]) # yellow

scene_center = scene_mesh.get_center()
box_center_1 = mesh_box_1.get_center()
box_center_2 = mesh_box_3.get_center()
mesh_box_1.translate((scene_center[0] - box_center_1[0], box_center_1[1] + scene_min_y, -1))
mesh_box_2.translate((scene_center[0] - box_center_1[0], box_center_1[1] + scene_max_y, -1))
mesh_box_3.translate((-box_center_2[0] + scene_max_x, scene_center[1] - box_center_2[1], -1))
mesh_box_4.translate((-box_center_2[0] + scene_min_x, scene_center[1] - box_center_2[1], -1))

R = scene_mesh.get_rotation_matrix_from_xyz((0, 0, -rot_angle_1))
mesh_box_1.rotate(R, center=(0,0,0))
mesh_box_2.rotate(R, center=(0,0,0))
mesh_box_3.rotate(R, center=(0,0,0))
mesh_box_4.rotate(R, center=(0,0,0))
humans = []
for i in range(8):
    body = o3d.io.read_triangle_mesh('human_meshes/' + str(i) + '.ply')
    body.compute_vertex_normals()
    humans.append(body)

# use normal open3d visualization
o3d.visualization.draw_geometries([mesh_box_1, mesh_box_2, mesh_box_3, mesh_box_4, scene_mesh] + humans)
