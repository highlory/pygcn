import numpy as np
import scipy.sparse as sp
import torch


# 对标签进行 one hot 编码
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    # 从文件中读取数据，得到一个二维数组，第一列为idx，最后一列为标签，中间的为paper的特征
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    # 生成特征
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # 生成标签的 one hot 向量表示
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)  #读取节点idx
    idx_map = {j: i for i, j in enumerate(idx)}  #生成一个节点id到[0, len(idx)-1]的哈希表，用于将节点id转换为编号
    # 得到原始的边信息，保存在n*2的数组中，n为边的个数
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    # 利用哈希表，将节点id转换为编号
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix  获得对称矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 对特征矩阵进行标准化
    features = normalize(features)  # question: 这里为什么要对特征进行标准化
    
    # 对邻接矩阵进行标准化，在标准化之前先加上一个单位矩阵，构成自环
    # 这里对邻接矩阵进行标准化，相当于公式中的给邻接矩阵左右乘上度矩阵的-1/2次方
    # 因为如果不对邻接矩阵进行标准化的话，和特征矩阵相乘的时候会改变特征原本的分布，产生不可预测的问题
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    # feature.todense()将 scipy.sparse.csr.csr_matrix 类型转换为 numpy.matrix 类型
    # 后面又紧接着转换为 numpy.array类型
    # 然后再转换为tensor类型
    features = torch.FloatTensor(np.array(features.todense()))
    # np.where(labels)相当于np.where(labels != 0),即寻找labels中元素不为0的坐标
    # 因为labels为二维数组，所以返回结果为长度为2的元组，元组的每一项为一个数组，保存该维上的坐标。
    # 因为每一行只有一个为1，其余为0，所以这里的实际作用为将类别的one hot编码变为类别id
    labels = torch.LongTensor(np.where(labels)[1])
    # 将邻接矩阵转换为tensor类型
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    # print('adj type:')
    # print(type(adj))
    # print('adj shape')
    # print(adj.size())

    # 将训练集、验证集和测试集的id转换为tensor类型
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    print(labels)
    print(labels.shape)

    return adj, features, labels, idx_train, idx_val, idx_test


# 对矩阵的每一行做标准化，标准化结果为每一行的行和为1（在本来行和不为0的前提下）
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

if __name__ == "__main__":
    load_data()
