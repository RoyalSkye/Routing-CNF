import os
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.distance import pdist, squareform
from sklearn.utils import shuffle
    
    
def get_DTSP_training_data():
    train = TSPDataset('/TSP20/train')
    val = TSPDataset('/TSP20/val')
    test = TSPDataset('/TSP20/test')
    return train, val, test, "DTSP-GEN"


class TSPDataset(Dataset):
    def __init__(self, path):
        self.data_names = [path + '/' + x for x in os.listdir(path)]
        self.path = path

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, idx):
        item = read_graph(self.data_names[idx])
        return item


def read_graph(filepath):
    with open(filepath, "r") as f:

        line = ''

        # Parse number of vertices
        while 'DIMENSION' not in line: line = f.readline();
        n = int(line.split()[1])
        Ma = np.zeros((n, n), dtype=int)
        Mw = np.zeros((n, n), dtype=float)

        # Parse edges
        while 'EDGE_DATA_SECTION' not in line: line = f.readline();
        line = f.readline()
        while '-1' not in line:
            i, j = [int(x) for x in line.split()]
            Ma[i, j] = 1
            line = f.readline()

        # Parse edge weights
        while 'EDGE_WEIGHT_SECTION' not in line: line = f.readline();
        for i in range(n):
            Mw[i, :] = [float(x) for x in f.readline().split()]

        # Parse tour
        while 'TOUR_SECTION' not in line: line = f.readline();
        route = [int(x) for x in f.readline().split()]
        
        # Parse tour
        while 'NODE_POSITION' not in line: line = f.readline();
        nodes = [x for x in f.readline().split()]

    return Ma, Mw, route, nodes, filepath


def get_CTSP_training_data():
    train = "../data/ConvTSP/TSP100/tsp100_train_concorde.txt"
    val = "../data/ConvTSP/TSP100/tsp100_val_concorde.txt"
    test = "../data/ConvTSP/TSP100/tsp100_test_concorde.txt"
    return train, val, test, "TSP100"


class DotDict(dict):
    """Wrapper around in-built dict class to access members through the dot operation.
    """

    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class GoogleTSPReader(object):
    """Iterator that reads TSP dataset files and yields mini-batches.
    
    Format expected as in Vinyals et al., 2015: https://arxiv.org/abs/1506.03134, http://goo.gl/NDcOIG
    """

    def __init__(self, num_nodes, num_neighbors, batch_size, filepath, changing_size=False):
        """
        Args:
            num_nodes: Number of nodes in TSP tours
            num_neighbors: Number of neighbors to consider for each node in graph
            batch_size: Batch size
            filepath: Path to dataset file (.txt file)
        """
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.filepath = filepath
        if changing_size:
            self.filedata = open(filepath, "r").readlines()
        else:
            self.filedata = shuffle(open(filepath, "r").readlines())  # Always shuffle upon reading data
        self.max_iter = (len(self.filedata) // batch_size)
        self.changing_size = changing_size

    def __iter__(self):
        for batch in range(self.max_iter):
            start_idx = batch * self.batch_size
            end_idx = (batch + 1) * self.batch_size
            yield self.process_batch(self.filedata[start_idx:end_idx], batch)

    def process_batch(self, lines, bid):
        """Helper function to convert raw lines into a mini-batch as a DotDict.
        """
        batch_edges = []
        batch_edges_values = []
        batch_edges_target = []  # Binary classification targets (0/1)
        batch_nodes = []
        batch_nodes_target = []  # Multi-class classification targets (`num_nodes` classes)
        batch_nodes_coord = []
        batch_tour_nodes = []
        batch_tour_len = []
        batch_opt_sol = []

        for line_num, line in enumerate(lines):
            line = line.split(" ")  # Split into list
            
            if self.changing_size:
                nn = self.num_nodes[line_num+bid]
            else:
                nn = self.num_nodes
            
            # Compute signal on nodes
            nodes = np.ones(nn)
            
            # Convert node coordinates to required format
            nodes_coord = []
            for idx in range(0, 2 * nn, 2):
                nodes_coord.append([float(line[idx]), float(line[idx + 1])])
            
            # Compute distance matrix
            W_val = squareform(pdist(nodes_coord, metric='euclidean'))
            
            # Compute adjacency matrix
            if self.num_neighbors == -1:
                W = np.ones((nn, nn))  # Graph is fully connected
            else:
                W = np.zeros((nn, nn))
                # Determine k-nearest neighbors for each node
                knns = np.argpartition(W_val, kth=self.num_neighbors, axis=-1)[:, self.num_neighbors::-1]
                # Make connections 
                for idx in range(nn):
                    W[idx][knns[idx]] = 1
            np.fill_diagonal(W, 2)  # Special token for self-connections
            
            # Convert tour nodes to required format
            # Don't add final connection for tour/cycle
            tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]][:-1]
            batch_opt_sol.append(tour_nodes)
            
            # Compute node and edge representation of tour + tour_len
            tour_len = 0
            nodes_target = np.zeros(nn)
            edges_target = np.zeros((nn, nn))
            for idx in range(len(tour_nodes) - 1):
                i = tour_nodes[idx]
                j = tour_nodes[idx + 1]
                nodes_target[i] = idx  # node targets: ordering of nodes in tour
                edges_target[i][j] = 1
                edges_target[j][i] = 1
                tour_len += W_val[i][j]
            
            # Add final connection of tour in edge target
            nodes_target[j] = len(tour_nodes) - 1
            edges_target[j][tour_nodes[0]] = 1
            edges_target[tour_nodes[0]][j] = 1
            tour_len += W_val[j][tour_nodes[0]]
            
            # Concatenate the data
            batch_edges.append(W)  # adjacency matrix
            batch_edges_values.append(W_val)  # distance matrix
            batch_edges_target.append(edges_target)
            batch_nodes.append(nodes)
            batch_nodes_target.append(nodes_target)
            batch_nodes_coord.append(nodes_coord)
            batch_tour_nodes.append(tour_nodes)
            batch_tour_len.append(tour_len)
        
        # From list to tensors as a DotDict
        batch = DotDict()
        batch.edges = np.stack(batch_edges, axis=0)
        batch.edges_values = np.stack(batch_edges_values, axis=0)
        batch.edges_target = np.stack(batch_edges_target, axis=0)
        batch.nodes = np.stack(batch_nodes, axis=0)
        batch.nodes_target = np.stack(batch_nodes_target, axis=0)
        batch.nodes_coord = np.stack(batch_nodes_coord, axis=0)
        batch.tour_nodes = np.stack(batch_tour_nodes, axis=0)
        batch.tour_len = np.stack(batch_tour_len, axis=0)

        return batch
