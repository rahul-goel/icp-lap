# Iterative Closest Point and Laplacian Mesh Editing

Name - Rahul Goel
Roll Number - 2019111034

## Iterative Closest Point
Run `python icp.py [-h] [--method METHOD] [--num NUM] [--trans TRANS] [--iter ITER] [--kd-tree | --no-kd-tree] [--spot | --bunny]`.

- `method` can be `svd` for Singular Value Decomposition, `lsq_point` for least squares optimization with point to point distance or `lsq_plane` for least squares optimization with point to plane distance.
- `spot` or `bunny` chooses the underlying geometry. For bunny, `num` can be set to sample any number of points on the mesh.
- `trans` sets the maximum possible translation in the random transformation.
- `kd-tree` can be used to accelerate nearest neighbor search.

## Laplacian Mesh Editing
Run `python mesh_editing.py` to display the original `spot` mesh and the later display the edited mesh.
The point to move and its new position and the control points defining the boundary have been hard-coded.
