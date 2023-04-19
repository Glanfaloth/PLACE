import warnings
warnings.simplefilter("ignore", UserWarning)
import argparse
import open3d as o3d
import os
import re
from models.cvae import *
from preprocess.preprocess_optimize import *
from preprocess.bps_encoding import *
from utils import *
from utils_read_data import *

prox_dataset_path = '/local/home/yeltao/Desktop/3DHuman_gen/dataset/proxe'
parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str, default="N3OpenArea")
args = parser.parse_args()
scene_name = args.scene

scene_directory = "scene_meshes/{}".format(args.scene)
human_directory = "human_meshes/{}".format(args.scene)
directory_pattern =  r'.*{}$'.format(args.scene)

if not os.path.exists(scene_directory):
    os.mkdir(scene_directory)
if not os.path.exists(human_directory):
    os.mkdir(human_directory)

num_instances = 0

for dirpath, dirnames, filenames in os.walk('/local/home/yeltao/Downloads'):
    for dirname in dirnames:
        if re.match(directory_pattern, dirname):
            directory_path = os.path.join(dirpath, dirname, "pred_mask")
            for filename in os.listdir(directory_path):
                num_instances += 1

# read scen mesh/sdf
scene_mesh, cur_scene_verts, s_grid_min_batch, s_grid_max_batch, s_sdf_batch = read_mesh_sdf(prox_dataset_path,
                                                                                             'prox',
                                                                                             scene_name)
rot_angle_1, scene_min_x, scene_max_x, scene_min_y, scene_max_y = define_scene_boundary('prox', scene_name)

scene_center = scene_mesh.get_center()
R = scene_mesh.get_rotation_matrix_from_xyz((0, 0, rot_angle_1))

T = np.zeros((4, 4))
T[0, 0] = -1
T[1, 2] = 1
T[2, 1] = 1
T[3, 3] = 1

num_humans = 8

humans = []
humans_t = []
bb = []
for i in range(num_humans):
    body = o3d.io.read_triangle_mesh(human_directory+'/' + str(i) + '.obj')
    body.compute_vertex_normals()
    body.rotate(R, center=(0,0,0))
    humans.append(body)
    body_t = copy.deepcopy(body).transform(T)
    humans_t.append(body_t)
    b = body_t.get_axis_aligned_bounding_box()
    bb.append(b)
    # move body to scene center
    human_center_t = body_t.get_center()
    body_centered = copy.deepcopy(body_t).translate(-human_center_t)
    o3d.io.write_triangle_mesh(human_directory+'/' + args.scene + '_human_'+ str(i) + "_centered.obj", body_centered, write_triangle_uvs=True)

    minCoors = [float(co) for co in b.get_print_info().split(") - (")[0][2:].split(", ")]
    maxCoors = [float(co) for co in b.get_print_info().split(") - (")[1][:-2].split(", ")]

    with open('/local/home/yeltao/thesis_ws/sb_min_time_quadrotor_planning/python/obstacles.txt', 'a', newline='') as file:
        file.write(" ".join(str(x) for x in [-maxCoors[0],minCoors[2],minCoors[1],-minCoors[0],maxCoors[2],maxCoors[1]])+ '\n')

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[0,0,0])

scene_mesh.rotate(R, center=(0,0,0))

# use normal open3d visualization

for idx in range(num_instances):
    instance_obj = o3d.io.read_triangle_mesh(scene_directory + '/' + args.scene +'_' + str(idx) + ".obj")
    instance_obj.rotate(R, center=(0,0,0))
    instance_t = copy.deepcopy(instance_obj).transform(T)
    b = instance_t.get_axis_aligned_bounding_box()
    bb.append(b)

    minCoors = [float(co) for co in b.get_print_info().split(") - (")[0][2:].split(", ")]
    maxCoors = [float(co) for co in b.get_print_info().split(") - (")[1][:-2].split(", ")]
    with open('/local/home/yeltao/thesis_ws/sb_min_time_quadrotor_planning/python/obstacles.txt', 'a', newline='') as file:
        file.write(" ".join(str(x) for x in [-maxCoors[0],minCoors[2],minCoors[1],-minCoors[0],maxCoors[2],maxCoors[1]])+ '\n')

world_min_x = 100
world_min_y = 100
world_min_z = 100
world_max_x = -100
world_max_y = -100
world_max_z = -100
for bb_len in range(len(bb)):
    minCoors = [float(co) for co in bb[bb_len].get_print_info().split(") - (")[0][2:].split(", ")]
    maxCoors = [float(co) for co in bb[bb_len].get_print_info().split(") - (")[1][:-2].split(", ")]
    world_min_x = min(world_min_x, -maxCoors[0])
    world_min_y = min(world_min_y, minCoors[2])
    world_min_z = min(world_min_z, minCoors[1])
    world_max_x = max(world_max_x, -minCoors[0])
    world_max_y = max(world_max_y, maxCoors[2])
    world_max_z = max(world_max_z, maxCoors[1])

object_name = "Object{}".format(num_humans+1)
with open('/local/home/yeltao/thesis_ws/sb_min_time_quadrotor_planning/python/obstacles.txt', 'a', newline='') as file:
        file.write(" ".join(str(x) for x in [world_min_x, world_min_y, world_min_z, world_max_x, world_max_y, world_max_z])+ '\n')
