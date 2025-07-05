import numpy as np
import scipy.sparse as sp
import pickle
import pandas as pd
import csv
import torch
from torch_geometric.data import Data
from .wrapper import sample_random_walks
def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None,length=50,sample_rate=1.,backtracking=False,strict=False,pad_idx=-1,window_size=8):
    if 'npy' in distance_df_filename:
        A = np.load(distance_df_filename)
    else:
        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)
        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)
        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
                    A = A + np.eye(A.shape[0])
        else:
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
                    A = A + np.eye(A.shape[0])
    adj_mx=torch.tensor(A)
    start_nodes, end_nodes = torch.nonzero(adj_mx, as_tuple=True)
    edges = torch.stack([start_nodes, end_nodes])
    walk_node_index, walk_edge_index, walk_node_id_encoding, walk_node_adj_encoding = sample_random_walks(
        edges,
        num_of_vertices,
        length,
        sample_rate,
        backtracking,
        strict,
        window_size,
        pad_idx
    )

    walk_node_idx = torch.from_numpy(walk_node_index)
    walk_edge_idx = torch.from_numpy(walk_edge_index)
    walk_node_id_encoding = torch.from_numpy(walk_node_id_encoding).float()
    walk_node_adj_encoding = torch.from_numpy(walk_node_adj_encoding).float()

    walk_pe = torch.cat([walk_node_id_encoding, walk_node_adj_encoding], dim=-1)
    '''
    walk_pe_dim = 8* 2 - 1
    self.walk_pe_proj = nn.Linear(walk_pe_dim, hidden_size)
    self.walk_pe_proj(walk_pe)
    '''

    return A,walk_node_idx, walk_edge_idx, walk_pe

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data



def calculate_scaled_laplacian(adj):
    """
    L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    L' = 2L/lambda - I

    Args:
        adj: adj_matrix

    Returns:
        np.ndarray: L'
    """
    n = adj.shape[0]
    d = np.sum(adj, axis=1)  # D
    lap = np.diag(d) - adj     # L=D-A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                lap[i, j] /= np.sqrt(d[i] * d[j])
    lap[np.isinf(lap)] = 0
    lap[np.isnan(lap)] = 0
    lam = np.linalg.eigvals(lap).max().real
    return 2 * lap / lam - np.eye(n)



def weight_matrix(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
    '''
    From STGCN-IJCAI2018
    Load weight matrix function.
    :param file_path: str, the path of saved weight matrix file.
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray, [n_route, n_route].
    '''
    try:
        W = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    # check whether W is a 0/1 matrix.
    if set(np.unique(W)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling:
        n = W.shape[0]
        W = W / 10000.
        W2, WMASK = W * W, np.ones([n, n]) - np.identity(n)
        # refer to Eq.10
        A = np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * WMASK
        return A
    else:
        return W


def first_approx(W, n):
    '''
    1st-order approximation function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, n_route].
    '''
    A = W + np.identity(n)
    d = np.sum(A, axis=1)
    sinvD = np.sqrt(np.mat(np.diag(d)).I)
    # refer to Eq.5
    return np.mat(np.identity(n) + sinvD * A * sinvD)

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def idEncode(x, y, col):
    return x * col + y

def constructGraph(row, col):
    mx = [-1, 0, 1, 0, -1, -1, 1, 1, 0]
    my = [0, -1, 0, 1, -1, 1, -1, 1, 0]

    areaNum = row * col

    def illegal(x, y):
        return x < 0 or y < 0 or x >= row or y >= col

    W = np.zeros((areaNum, areaNum))
    for i in range(row):
        for j in range(col):
            n1 = idEncode(i, j, col)
            for k in range(len(mx)):
                temx = i + mx[k]
                temy = j + my[k]
                if illegal(temx, temy):
                    continue
                n2 = idEncode(temx, temy, col)
                W[n1, n2] = 1
    return W


def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    isolated_point_num = np.sum(np.where(d, 0, 1))
    print(f"Number of isolated points: {isolated_point_num}")
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian, isolated_point_num

def cal_lape(adj_mx):
    lape_dim = 8
    L, isolated_point_num = calculate_normalized_laplacian(adj_mx)
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    laplacian_pe = EigVec[:, isolated_point_num + 1: lape_dim + isolated_point_num + 1]
    return laplacian_pe

def node_random_path(distance_df_filename, num_of_vertices,length=50,sample_rate=1.,backtracking=False,strict=False,pad_idx=-1,window_size=8):
    if 'adj.npy' in distance_df_filename:
        adj_mx = torch.tensor(np.load(distance_df_filename))
        start_nodes, end_nodes = torch.nonzero(adj_mx, as_tuple=True)
        edges = torch.stack([start_nodes, end_nodes])
    else:
        df = pd.read_csv(distance_df_filename, usecols=[0, 1])
        edges = torch.tensor(df.values.T, dtype=torch.float32)
    walk_node_index, walk_edge_index, walk_node_id_encoding, walk_node_adj_encoding = sample_random_walks(
        edges,
        num_of_vertices,
        length,
        sample_rate,
        backtracking,
        strict,
        window_size,
        pad_idx
    )
    walk_node_idx = torch.from_numpy(walk_node_index)
    walk_edge_idx = torch.from_numpy(walk_edge_index)
    walk_node_id_encoding = torch.from_numpy(walk_node_id_encoding).float()
    walk_node_adj_encoding = torch.from_numpy(walk_node_adj_encoding).float()
    walk_pe = torch.cat([walk_node_id_encoding, walk_node_adj_encoding], dim=-1)

    return walk_node_idx, walk_edge_idx, walk_pe


