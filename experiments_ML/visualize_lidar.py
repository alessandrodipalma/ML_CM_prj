import numpy as np
import open3d
from open3d.cpu.pybind.io import read_point_cloud
from open3d.cpu.pybind.visualization import draw_geometries

def main():

    cloud = read_point_cloud("cluster_0.pcd") # Read the point cloud
    draw_geometries([cloud]) # Visualize the point cloud
    open3d.geometry.create_surface_voxel_grid_from_point_cloud(cloud)
    print(cloud)
    # for pt in cloud:

        # point_xy_greyscale.x < - pt.x
        # point_xy_greyscale.y < - pt.y
        # point_xy_greyscale.greyscale < - map(pt.z, z_min, z_max, 0, 255)
        # greyscale_vector.add(point_xy_greyscale)


if __name__ == "__main__":
    main()