import numpy as np
import argparse
import scipy
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
from time import time

args = None

def get_pc(args):
    if args.bunny:
        mesh = o3d.io.read_triangle_mesh("./bunny.ply")
        pc = mesh.sample_points_uniformly(number_of_points=args.num)
    elif args.spot:
        pc = o3d.io.read_point_cloud("./spot.xyz")
    return pc

def random_transform(pc, trans):
    points = np.array(pc.points)
    r = R.from_euler("XYZ", np.random.uniform(0, 45, size=3), degrees=True).as_matrix()
    r_points = (r @ points.T).T
    t = np.random.uniform(-trans, trans, 3)
    tr_points = r_points + t
    pc.points = o3d.utility.Vector3dVector(tr_points)
    return pc

def svd_icp(pc_a_full, pc_b_full, pc_b):
    a = np.array(pc_a_full.points)
    b = np.array(pc_b.points)
    am, bm = np.array(pc_a_full.points).mean(axis=0), np.array(pc_b_full.points).mean(axis=0)
    am, bm = am.reshape(1, 3), bm.reshape(1, 3)
    ap, bp = a - am, b - bm
    cov = ap.T @ bp

    u, _, vt = np.linalg.svd(cov)

    r = u @ vt
    if not np.isclose(np.linalg.det(r), 1.0):
        r = u @ np.diag([1, 1, -1]) @ vt
    t = np.array(pc_b_full.points).mean(axis=0).reshape(1, 3) - np.array(pc_a_full.points).mean(axis=0).reshape(1, 3) @ r

    a_full = np.array(pc_a_full.points)
    a_full = a_full @ r + t
    pc_a_full.points = o3d.utility.Vector3dVector(a_full)

    return pc_a_full

def least_squares_icp(pc_a_full, pc_b_full, idx, args):
    a, b = np.array(pc_a_full.points), np.array(pc_b_full.points)
    params = np.random.uniform(size=7)

    def loss_point(params):
        r = R.from_quat(params[:4]).as_matrix()
        t = params[4:].reshape(1, 3)
        return np.sum((a @ r + t - b[idx]) ** 2, axis=1)

    def loss_plane(params):
        r = R.from_quat(params[:4]).as_matrix()
        t = params[4:].reshape(1, 3)
        bn = np.array(pc_b_full.normals)[idx]
        p = (a @ r + t) - b[idx]
        d = np.sum(p * bn, axis=1)
        return d

    if args.method == "lsq_point":
        sol = scipy.optimize.least_squares(loss_point, params)
    elif args.method == "lsq_plane":
        sol = scipy.optimize.least_squares(loss_plane, params)

    params = sol.x
    r = R.from_quat(params[:4]).as_matrix()
    t = params[4:].reshape(1, 3)

    a_full = np.array(pc_a_full.points)
    a_full = a_full @ r + t
    pc_a_full.points = o3d.utility.Vector3dVector(a_full)

    return pc_a_full

def main(args):
    # a moves, b is fixed

    a = get_pc(args)
    a.paint_uniform_color(np.array([1, 0, 0]))
    b = random_transform(deepcopy(a), args.trans)
    b.paint_uniform_color(np.array([0, 0, 1]))

    cb = deepcopy(b)

    o3d.visualization.draw_geometries([a, b])

    start_time = time()

    if args.kd_tree:
        kd_tree = scipy.spatial.KDTree(np.array(b.points) - np.array(b.points).mean(axis=0))

    if args.method == "lsq_plane":
        b.estimate_normals()

    for _ in range(args.iter):
        if args.kd_tree:
            _, idx = kd_tree.query(np.array(a.points) - np.array(a.points).mean(axis=0))
        else:
            idx = np.argmin(scipy.spatial.distance.cdist(np.array(a.points) - np.array(a.points).mean(axis=0), np.array(b.points) - np.array(b.points).mean(axis=0)), axis=1)


        if args.method == "svd":
            cb.points = o3d.utility.Vector3dVector(np.array(b.points)[idx])
            a = svd_icp(a, b, cb)
        elif "lsq" in args.method:
            a = least_squares_icp(a, b, idx, args)

        print("\r", np.sum(np.linalg.norm(np.array(a.points) - np.array(b.points))), end=' ' * 10)
    print("\n")
    o3d.visualization.draw_geometries([a, b])
    print(np.isclose(np.array(a.points), np.array(b.points)).all())

    end_time = time()
    print("Time taken:", end_time - start_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--method", type=str, default="svd")
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--trans", type=float, default=100.0)
    parser.add_argument("--iter", type=int, default=100)
    parser.add_argument("--kd-tree", action=argparse.BooleanOptionalAction)
    parser.add_argument("--spot", action=argparse.BooleanOptionalAction)
    parser.add_argument("--bunny", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    main(args)
