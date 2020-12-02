import torch.nn as nn
import torch
from engineer.models.registry import BACKBONES
from engineer.models.common.helper import *
from engineer.models.common.semgcn_helper import _ResGraphConv_Attention,SemGraphConv,_GraphConv
from engineer.models.common.HM import HM_Extrect
from scipy import sparse as sp
import numpy as np



@BACKBONES.register_module
class SemGCN_Attention(nn.Module):
    def __init__(self,adj,num_joints, hid_dim, coords_dim=(2, 2), p_dropout=None):
        '''
        :param adj:  adjacency matrix using for
        :param hid_dim:
        :param coords_dim:
        :param num_layers:
        :param nodes_group:
        :param p_dropout:
        '''

        super(SemGCN_Attention, self).__init__()


        self.heat_map_head =[]

        self.gcn_head=[]
        self.generator_map=[]


        self.heat_map_generator =HM_Extrect(12)

        self.heat_map_head.append(self.heat_map_generator)
        self.adj = self._build_adj_mx_from_edges(num_joints,adj)
        adj = self.adj_matrix



        self.gconv_input = _GraphConv(adj, coords_dim[0], hid_dim[0], p_dropout=p_dropout)
        # in here we set 4 gcn model in this part
        self.gconv_layers1 = _ResGraphConv_Attention(adj, hid_dim[0], hid_dim[1], hid_dim[0], p_dropout=p_dropout)
        self.gconv_layers2 = _ResGraphConv_Attention(adj, hid_dim[1]+256, hid_dim[2]+256, hid_dim[1]+256, p_dropout=p_dropout)
        self.gconv_layers3 = _ResGraphConv_Attention(adj, hid_dim[2]+384, hid_dim[3]+384, hid_dim[2]+384, p_dropout=p_dropout)
        self.gconv_layers4 = _ResGraphConv_Attention(adj, hid_dim[3]+512, hid_dim[4]+512, hid_dim[3]+512, p_dropout=p_dropout)


        self.gconv_output1 = SemGraphConv(384, coords_dim[1], adj)
        self.gconv_output2 = SemGraphConv(512, coords_dim[1], adj)
        self.gconv_output3 = SemGraphConv(640, coords_dim[1], adj)


        self.gcn_head.append(self.gconv_input)
        self.gcn_head.append(self.gconv_layers1)
        self.gcn_head.append(self.gconv_layers2)
        self.gcn_head.append(self.gconv_layers3)
        self.gcn_head.append(self.gconv_layers4)
        self.gcn_head.append(self.gconv_output1)
        self.gcn_head.append(self.gconv_output2)
        self.gcn_head.append(self.gconv_output3)


    def extract_features_joints(self,ret_features,hms):
        '''
        extract features from joint feature_map

        :return:
        '''

        joint_features = []


        for feature, hm_pred in zip(ret_features, hms):
            joint_feature = torch.zeros([feature.shape[0], feature.shape[1], hm_pred.shape[1]]).cuda()
            for bz in range(feature.shape[0]):
                for joint in range(hm_pred.shape[1]):
                    joint_feature[bz, :, joint] = feature[bz, :, hm_pred[bz, joint, 1], hm_pred[bz, joint, 0]]
            joint_features.append(joint_feature)
        return joint_features
    @property
    def adj_matrix(self):
        return self.adj

    @adj_matrix.setter
    def adj_matrix(self,adj_matrix):
        self.adj = adj_matrix


    def _build_adj_mx_from_edges(self,num_joints,edge):
        def adj_mx_from_edges(num_pts, edges, sparse=True):
            edges = np.array(edges, dtype=np.int32)
            data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
            adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

            # build symmetric adjacency matrix
            adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
            adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
            if sparse:
                adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
            else:
                adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
            return adj_mx

        def sparse_mx_to_torch_sparse_tensor(sparse_mx):
            """Convert a scipy sparse matrix to a torch sparse tensor."""
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
            indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
            values = torch.from_numpy(sparse_mx.data)
            shape = torch.Size(sparse_mx.shape)
            return torch.sparse.FloatTensor(indices, values, shape)

        def normalize(mx):
            """Row-normalize sparse matrix"""
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sp.diags(r_inv)
            mx = r_mat_inv.dot(mx)
            return mx

        return adj_mx_from_edges(num_joints, edge, False)


    def forward(self, x,hm_4,ret_features):


        results,heat_map = self.heat_map_generator(ret_features)
        heat_map_intergral,score = softmax_integral_tensor(heat_map,12,heat_map.shape[-1],heat_map.shape[-2])

        hm_4 = heat_map_intergral.view(-1,12,2)
        score = score.view(-1,12,1)

        j_1_16 = F.grid_sample(results[0],hm_4[:,None,:,:]).squeeze(2)
        j_1_8 = F.grid_sample(results[1],hm_4[:,None,:,:]).squeeze(2)
        j_1_4 = F.grid_sample(results[2],hm_4[:,None,:,:]).squeeze(2)

        # x = torch.cat([hm_4,score],-1)
        out = self.gconv_input(x)
        #gconv_layers in here is residual GCN.
        #label == 0
        out = self.gconv_layers1(out,None)
        out = self.gconv_layers2(out,j_1_16)
        out1 = self.gconv_output1(out)

        out = self.gconv_layers3(out,j_1_8)
        out2 = self.gconv_output2(out)

        out = self.gconv_layers4(out,j_1_4)
        out3 = self.gconv_output3(out)

        return [out1,out2,out3],heat_map_intergral,x