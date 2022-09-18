import numpy as np
import scipy
import open3d as o3d
import networkx as nx
from copy import deepcopy

def get_mesh_edges(mesh):
    triangles = np.array(mesh.triangles)
    edges_0 = np.vstack([triangles[:, 0], triangles[:, 1]]).T
    edges_1 = np.vstack([triangles[:, 1], triangles[:, 2]]).T
    edges_2 = np.vstack([triangles[:, 2], triangles[:, 0]]).T
    edges = np.vstack([edges_0, edges_1, edges_2])
    return edges

def get_graph(vertices, edges):
    g = nx.Graph()
    vertex_data = [(i, {"position": vertices[i]}) for i in range(len(vertices))]
    g.add_nodes_from(vertex_data)
    g.add_edges_from(edges)
    return g

def get_boundary(g, control_points):
    boundary = []
    num = len(control_points)
    for i in range(num):
        v1, v2 = control_points[i], control_points[(i + 1) % num]
        path = nx.shortest_path(g, v1, v2)
        boundary += path[0:-1]
    return boundary

def get_change_points(g, boundary, handle):
    g_dash = g.copy()
    g_dash.remove_nodes_from(boundary)
    points = nx.node_connected_component(g_dash, handle)
    return list(points)

def get_laplacian(g):
    adj = np.array(nx.adjacency_matrix(g).todense())
    degrees = np.sum(adj, axis=0)
    deg = np.diag(degrees)
    inv_deg = np.linalg.pinv(deg)
    return np.identity(inv_deg.shape[0]) - inv_deg @ adj

def get_neighbor(g, n, l2g, g2l):
    nb = []
    for i in g.neighbors(l2g[n]):
        nb.append(g2l[i])
    return nb

def pick_vertices(mesh):
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()
    picked_points = vis.get_picked_points()
    points = [point.index for point in picked_points]
    print(points)

def main():
    mesh = o3d.io.read_triangle_mesh("./spot.obj")
    handle = 2034
    final_pos = [-0.67, 0.90, -0.10]
    control_points = [2018, 2453, 3177]

    mesh.compute_vertex_normals()
    vertices = np.array(mesh.vertices)
    edges = get_mesh_edges(mesh)

    g = get_graph(vertices, edges)
    boundary = get_boundary(g, control_points)
    change_points = get_change_points(g, boundary, handle)

    sg = g.subgraph(boundary + change_points)
    g2l = {}
    for i in sg.nodes:
        g2l[i] = len(g2l)
    l2g = list(sg.nodes)

    L = get_laplacian(sg)
    V = np.array([sg.nodes[i]['position'] for i in sg.nodes])
    delta = L @ V
    n = delta.shape[0]

    LS = np.zeros([3*n, 3*n])
    for i in range(3):
        LS[i*n:(i+1)*n,i*n:(i+1)*n] = -L

    for i in range(n):
        nb_idx = get_neighbor(sg, i, l2g, g2l)
        ring = np.array([i] + nb_idx)
        vring = V[ring]
        n_ring = vring.shape[0]
        
        A = np.zeros([n_ring * 3, 7])
        for j in range(n_ring):
            A[j]          = [vring[j][0],            0,   vring[j][2], -vring[j][1], 1, 0, 0]
            A[j+n_ring]   = [vring[j][1], -vring[j][2],            0 ,  vring[j][0], 0, 1, 0]
            A[j+2*n_ring] = [vring[j][2],  vring[j][1], -vring[j][0],             0, 0, 0, 1]
            

        ainv = np.linalg.pinv(A)
        s = ainv[0]
        h = ainv[1:4]
        _ = ainv[4:7]
        

        tdelta = np.vstack([
             delta[i][0]*s    - delta[i][1]*h[2] + delta[i][2]*h[1],
             delta[i][0]*h[2] + delta[i][1]*s    - delta[i][2]*h[0],
            -delta[i][0]*h[1] + delta[i][1]*h[0] + delta[i][2]*s   ,
        ])

        LS[i, np.hstack([ring, ring+n, ring+2*n])] += tdelta[0]
        LS[i+n, np.hstack([ring, ring+n, ring+2*n])] += tdelta[1]
        LS[i+2*n, np.hstack([ring, ring+n, ring+2*n])] += tdelta[2]


    constraint_coef = []
    constraint_b = []

    boundary_idx = [g2l[i] for i in control_points]
    for idx in boundary_idx:
        for i in range(3):
            constraint_coef.append(np.arange(3*n) == idx + i*n)
            constraint_b.append(V[idx, i])
        
    idx = g2l[handle]
    pos = final_pos
    for i in range(3):
        constraint_coef.append(np.arange(3*n) == idx + i*n)
        constraint_b.append(pos[i])
        
    constraint_coef = np.matrix(constraint_coef)
    constraint_b = np.array(constraint_b)

    A = np.vstack([LS, constraint_coef])
    A = scipy.sparse.coo_matrix(A)
    b = np.hstack([np.zeros(3*n), constraint_b])

    vdash = scipy.sparse.linalg.lsqr(A, b)

    new_pnts = []
    for i in range(n):
        new_pnts.append([vdash[0][i], vdash[0][i+n], vdash[0][i+2*n]])
        
    new_mesh = deepcopy(mesh)
    for idx, pnt in enumerate(new_pnts):
        gid = l2g[idx]
        new_mesh.vertices[gid] = pnt

    o3d.io.write_triangle_mesh("newmesh.ply", new_mesh)

    o3d.visualization.draw_geometries([mesh])
    o3d.visualization.draw_geometries([new_mesh])
     
if __name__ == "__main__":
    main()
