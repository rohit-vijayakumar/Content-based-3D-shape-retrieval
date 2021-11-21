import open3d as o3d
import numpy as np

# examples/Python/Basic/file_io.py

import open3d as o3d

if __name__ == "__main__":

    print("Testing IO for meshes ...")
    mesh = o3d.io.read_triangle_mesh("m671.off")
    print(mesh)
    o3d.io.write_triangle_mesh("copy_of_knot.ply", mesh)

    print("Testing IO for textured meshes ...")
    textured_mesh = o3d.io.read_triangle_mesh("m671.off")
    print(textured_mesh)
    o3d.io.write_triangle_mesh("copy_of_crate.obj",
                               textured_mesh,
                               write_triangle_uvs=True)
    copy_textured_mesh = o3d.io.read_triangle_mesh('copy_of_crate.obj')
    print(copy_textured_mesh)



print("Try to render a mesh with normals (exist: " +
        str(mesh.has_vertex_normals()) + ") and colors (exist: " +
        str(mesh.has_vertex_colors()) + ")")
o3d.visualization.draw_geometries([mesh])
print("A mesh with no normals and no colors does not seem good.")