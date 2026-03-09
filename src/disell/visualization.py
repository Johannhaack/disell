import numpy as np
import pyvista as pv
from skimage.measure import marching_cubes
import open3d as o3d
import vtk
import vtk.util.numpy_support as ns



def smooth_mesh(mesh, method="none", iterations=30):
    if method == "none":
        return mesh
    elif method == "laplacian":
        return mesh.filter_smooth_simple(number_of_iterations=iterations)
    elif method == "taubin":
        return mesh.filter_smooth_taubin(number_of_iterations=iterations)
    else:
        raise ValueError(f"Unknown smoothing method '{method}'")


def o3d_to_vtk(o3d_mesh, rgb_color, grain_id):
    verts = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)

    # Convert vertices
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vtk.util.numpy_support.numpy_to_vtk(verts.astype(np.float32)))

    # Convert faces
    vtk_faces = vtk.vtkCellArray()
    for f in faces:
        cell = vtk.vtkTriangle()
        for i in range(3):
            cell.GetPointIds().SetId(i, int(f[i]))
        vtk_faces.InsertNextCell(cell)

    # Create PolyData
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetPolys(vtk_faces)

    # Add RGB colors
    colors = np.tile(rgb_color, (len(verts), 1))
    vtk_colors = vtk.util.numpy_support.numpy_to_vtk(
        (colors * 255).astype(np.uint8),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR,
    )
    vtk_colors.SetName("RGB")
    vtk_colors.SetNumberOfComponents(3)  # ensure 3-component vector
    poly.GetPointData().SetScalars(vtk_colors)  # <-- mark as actual color scalars

    # Add grain id
    grain_ids = np.full(len(verts), grain_id, dtype=np.int32)
    vtk_grain_id = vtk.util.numpy_support.numpy_to_vtk(grain_ids)
    vtk_grain_id.SetName("grain_id")
    poly.GetPointData().AddArray(vtk_grain_id)

    return poly


def export_grain_meshes(seg_filled, rgb_vol, output_path="grains_surface.vtp", voxel_size = (1,1,1), selected_grains = None,
                        min_voxels=50, type="volume", smoothing="none", smoothing_iterations=30):
    unique_ids = np.unique(seg_filled)
    unique_ids = unique_ids[unique_ids != 0]


    if output_path != "grains_surface.vtp":
        #check if the path is .vtp
        file_type = output_path.split('.')[-1]
        if file_type != "vtp":
            raise TypeError("output path must specify the file type which needs to be .vtp")
    if selected_grains is not None:
        unique_ids = [gid for gid in unique_ids if gid in selected_grains]
    dz, dy, dx = voxel_size

    vtk_data = vtk.vtkAppendPolyData()
    grains_count = 0
    rgb_vol = np.transpose(rgb_vol, (0, 2, 1, 3))

    for grain_id in unique_ids:
        grain_mask = seg_filled == grain_id
        #switch y and x axis
        grain_mask = np.transpose(grain_mask, (0, 2, 1))
        if np.count_nonzero(grain_mask) < min_voxels:
            continue

        if type == "volume":
            padded = np.pad(grain_mask, 1, constant_values=0)
            verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=(dz, dy, dx))
            verts -= np.array([dz, dy, dx])
            mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(verts),
                triangles=o3d.utility.Vector3iVector(faces)
            )
        elif type == "points":
            zz, yy, xx = np.nonzero(grain_mask)
            coords = np.stack([xx*dx, yy*dy, zz*dz], axis=-1).astype(np.float32)
            mesh = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(coords))
        else:
            raise ValueError("type must be 'volume' or 'points'")

        mesh.compute_vertex_normals()
        mesh = smooth_mesh(mesh, method=smoothing, iterations=smoothing_iterations)

        rgb_vals = rgb_vol[grain_mask]
        mean_rgb = np.clip(np.nanmean(rgb_vals, axis=0), 0, 1)

        poly = o3d_to_vtk(mesh, mean_rgb, grain_id)
        vtk_data.AddInputData(poly)
        grains_count += 1	

    vtk_data.Update()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputConnection(vtk_data.GetOutputPort())
    writer.Write()
    print(f"✅ Saved {grains_count} grains → {output_path}")