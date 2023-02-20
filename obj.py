import bpy
import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str, default="MPH1Library")
args = parser.parse_args()

directory_pattern =  r'.*{}$'.format(args.scene)

num_files = 0

for dirpath, dirnames, filenames in os.walk('/local/home/yeltao/Downloads'):
    for dirname in dirnames:
        if re.match(directory_pattern, dirname):
            directory_path = os.path.join(dirpath, dirname, "pred_mask")
            for filename in os.listdir(directory_path):
                num_files += 1

for j in range(num_files):
    # Import PLY file
    ply_path = "/local/home/yeltao/thesis_ws/PLACE/scene_meshes/"+args.scene+"/" + args.scene +"_" + str(j) + ".ply"
    bpy.ops.import_mesh.ply(filepath=ply_path)

    # Export OBJ file
    obj_path = "/local/home/yeltao/thesis_ws/PLACE/scene_meshes/"+args.scene+"/" + args.scene +"_"+str(j)+".obj"
    bpy.ops.export_scene.obj(filepath=obj_path, use_selection=True)