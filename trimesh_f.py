import numpy as np
import trimesh
from trimesh import viewer
from trimesh.viewer import windowed
mesh = trimesh.load('m671.off', force='mesh')

# viewer(flags="wireframe")

print(type(mesh.vertices))
# mesh.show()


import glob
a = glob.glob(r"E:\Documenten E\University\Jaar 5\Blok 1\Multimedia Retrieval\assignment\benchmark\db\**\*.off",recursive=True)
print(a[0])
# mesh = trimesh.load("Shadowboy_Idle.obj",force='mesh')

mesh.show()