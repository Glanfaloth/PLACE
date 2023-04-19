import warnings
warnings.simplefilter("ignore", UserWarning)
import argparse
import open3d as o3d
import os
import re
import yaml
import csv
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
mesh_boxes = [mesh_box_1, mesh_box_2, mesh_box_3, mesh_box_4]
R = scene_mesh.get_rotation_matrix_from_xyz((0, 0, rot_angle_1))

T = np.zeros((4, 4))
T[0, 0] = -1
T[1, 2] = 1
T[2, 1] = 1
T[3, 3] = 1

dynamic_objects = {}
static_objects = []
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
    
    object_name = "Object{}".format(i+1)
    dynamic_objects[object_name] = {}
    dynamic_objects[object_name]["csvtraj"] = "traj_human_{}".format(i)
    dynamic_objects[object_name]["loop"] = False
    dynamic_objects[object_name]["position"] = [float(-human_center_t[0]),float(human_center_t[2]),float(human_center_t[1])]
    dynamic_objects[object_name]["prefab"] = args.scene + '_human_'+ str(i) + "_centered"
    dynamic_objects[object_name]["rotation"] = [0,0,0,0]
    dynamic_objects[object_name]["scale"] = [1,1,1]
    dynamic_objects[object_name]["boundingbox"] = [-maxCoors[0],minCoors[2],minCoors[1],-minCoors[0],maxCoors[2],maxCoors[1]]

    with open('/local/home/yeltao/thesis_ws/agile_flight/flightmare/flightpy/configs/vision/custom/environment_0/csvtrajs/' + "traj_human_{}.csv".format(i), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["# header"])
        writer.writerow(['0.000000000000000000e+00', str(float(-human_center_t[0])), str(float(human_center_t[2])), str(float(human_center_t[1])), '0', '0', '0', '0'])

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[0,0,0])

scene_mesh.rotate(R, center=(0,0,0))


scene_mesh_t = copy.deepcopy(scene_mesh).transform(T)
mesh_frame_t = copy.deepcopy(mesh_frame).transform(T)
mesh_box_1_t = copy.deepcopy(mesh_box_1).transform(T)
mesh_box_2_t = copy.deepcopy(mesh_box_1).transform(T)
mesh_box_3_t = copy.deepcopy(mesh_box_1).transform(T)
mesh_box_4_t = copy.deepcopy(mesh_box_1).transform(T)
mesh_boxes_t = [mesh_box_1_t, mesh_box_2_t, mesh_box_3_t, mesh_box_4_t]
scene_center_t = scene_mesh_t.get_center()

# use normal open3d visualization

scene_mesh_centered = copy.deepcopy(scene_mesh_t).translate(-scene_center_t)
o3d.io.write_triangle_mesh(human_directory+"/"+ args.scene +"_scene_mesh_centered.obj", scene_mesh_centered)
scene_meshes = [scene_mesh_t]
for idx in range(num_instances):
    instance_obj = o3d.io.read_triangle_mesh(scene_directory + '/' + args.scene +'_' + str(idx) + ".obj")
    instance_obj.rotate(R, center=(0,0,0))
    instance_t = copy.deepcopy(instance_obj).transform(T)
    instance_center_t = instance_t.get_center()
    instance_centered = copy.deepcopy(instance_t).translate(-instance_center_t)
    o3d.io.write_triangle_mesh(scene_directory + '/' + args.scene +'_scene_'  + str(idx) + "_centered.obj", instance_centered, write_triangle_uvs=True)
    scene_meshes.append(instance_t)
    b = instance_t.get_axis_aligned_bounding_box()
    bb.append(b)

    minCoors = [float(co) for co in b.get_print_info().split(") - (")[0][2:].split(", ")]
    maxCoors = [float(co) for co in b.get_print_info().split(") - (")[1][:-2].split(", ")]
    
    tmp = [args.scene + '_scene_'+ str(idx) + "_centered"]
    tmp.extend([float(-instance_center_t[0]),float(instance_center_t[2]),float(instance_center_t[1])])
    tmp.extend([0,0,0,0])
    tmp.extend([1.0,1.0,1.0])
    tmp.extend([-maxCoors[0],minCoors[2],minCoors[1],-minCoors[0],maxCoors[2],maxCoors[1]])
    static_objects.append(", ".join([str(x) for x in tmp]))

all_meshes = humans_t + scene_meshes + bb

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
dynamic_objects[object_name] = {}
dynamic_objects[object_name]["csvtraj"] = "traj_scene"
dynamic_objects[object_name]["loop"] = False
dynamic_objects[object_name]["position"] = [float(-scene_center_t[0]),float(scene_center_t[2]),float(scene_center_t[1])]
dynamic_objects[object_name]["prefab"] = args.scene + '_scene_mesh_centered'
dynamic_objects[object_name]["rotation"] = [0,0,0,0]
dynamic_objects[object_name]["scale"] = [1,1,1]
dynamic_objects[object_name]["boundingbox"] = [world_min_x, world_min_y, world_min_z, world_max_x, world_max_y, world_max_z]

with open('/local/home/yeltao/thesis_ws/agile_flight/flightmare/flightpy/configs/vision/custom/environment_0/csvtrajs/' + "traj_scene.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["# header"])
    writer.writerow(['0.000000000000000000e+00', str(float(-scene_center_t[0])), str(float(scene_center_t[2])), str(float(scene_center_t[1])), '0', '0', '0', '0'])


with open("/local/home/yeltao/thesis_ws/agile_flight/flightmare/flightpy/configs/vision/custom/environment_0/static_obstacles.csv", "w") as f:
    f.write("\n".join(static_objects))

result = {"N": num_humans + 1, **dynamic_objects}
with open("/local/home/yeltao/thesis_ws/agile_flight/flightmare/flightpy/configs/vision/custom/environment_0/dynamic_obstacles.yaml", 'w') as f:
    yaml.dump(result, f, default_flow_style=False)
o3d.visualization.draw_geometries(all_meshes)