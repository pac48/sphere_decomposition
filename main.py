import numpy as np
import sphere_decomposition.sphere_decomposition_py as sphere_decomposition_py
import time
import pywavefront


def decompose_into_triangles(faces, verts):
    i_vertices = faces[:, 0:1]
    j_vertices = faces[:, 1:2]
    k_vertices = faces[:, 2:3]
    i_coords = verts[i_vertices]
    j_coords = verts[j_vertices]
    k_coords = verts[k_vertices]
    triangles = np.stack([i_coords, j_coords, k_coords], axis=-2)
    triangles = np.reshape(triangles, (-1, 3))
    return triangles


if __name__ == "__main__":
    model = pywavefront.Wavefront('cube.obj', collect_faces=True)
    faces = []
    for mesh_name in model.meshes:
        mesh = model.meshes[mesh_name]
        faces.append(np.array(mesh.faces))
    faces = np.vstack(faces)
    verts = np.array(model.vertices)
    triangles = decompose_into_triangles(faces, verts)

    controller = sphere_decomposition_py.ImguiController()
    # triangles[:, 2] += -1

    while True:
        start = time.time()
        triangles[:, 0] += .05*np.sin(time.time())
        triangles[:, 1] += .05*np.cos(time.time())

        res_y = controller.get_height()
        res_x = controller.get_width()
        fy = 1
        fx = 1
        img = sphere_decomposition_py.render(fx, fy, res_x, res_y, triangles)
        img = img[:, :, :3]
        controller.set_img(img[::-1, :, :].copy())
        print(time.time()-start)
        # time.sleep(.01)
