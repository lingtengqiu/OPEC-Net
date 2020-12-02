import cv2
import sys
sys.path.append("./")
from torch.utils.data import Dataset
import json
import os
import numpy as np
import random
from engineer.SPPE.src.utils.img import cropBox, im_to_torch
from opt import opt
import torch
import scipy.sparse as sp
from .pipelines import Compose
from .registry import DATASETS
try:
    from utils.img import transformBox_batch
except ImportError:
    from engineer.SPPE.src.utils.img import transformBox_batch


#EDGE only 12 joints and 11 edge
alpha_pose_index = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',\
 'left_hip', 'right_hip', 'left_knee','right_knee','left_ankle','right_ankle', 'head', 'neck']
EDGE = ([0,1],[0,2],[2,4],[1,3],[3,5],[0,6],[1,7],[6,7],[6,8],[8,10],[7,9],[9,11])
flip_index = [1,0,3,2,5,4,7,6,9,8,11,10]



@DATASETS.register_module
class PoseDataset(Dataset):

    def __init__(self,json_file,img_dir,black_list,pipeline):

        super(PoseDataset,self).__init__()

        self.img_list = []
        self.yolo_list = []
        self.gts_list=[]
        self.dts_list=[]
        self.img_dir = img_dir
        self.black_list = black_list
        self.score_list= []
        self.data_type=[]
        self.pipeline = Compose(pipeline)
        self._load_annotations(json_file)

    def __getitem__(self, item):

        img_name = self.img_list[item]
        types = self.data_type[item]
        orig_img = os.path.join(self.img_dir[types], img_name)
        orig_img = [cv2.imread(orig_img)]
        dts = self.dts_list[item]
        gts = self.gts_list[item]
        scores = torch.from_numpy(np.asarray(self.score_list[item])).float().view(-1, 1)
        yolos = torch.from_numpy(np.asarray(self.yolo_list[item])).float()
        inps = torch.zeros(yolos.size(0), 3, opt.inputResH, opt.inputResW)
        pt1 = torch.zeros(yolos.size(0), 2)
        pt2 = torch.zeros(yolos.size(0), 2)
        assert self.pipeline is not None

        result = dict(orig_img = orig_img[0], im_name=img_name, boxes=yolos,scores=scores, inps=inps, pt1=pt1, pt2=pt2)
        result = self.pipeline(result)
        result['gts'] = gts
        result['dts'] = dts
        return result
        #look wehter it need flip

    def __coco_name(self,img_name):
        pre__ = ""
        for i in range(16 - len(img_name)):
            pre__ += '0'
        img_name = pre__ + img_name
        return img_name
    def __load_annotations(self,json_info,type):
        with open(json_info,'r') as json_reader:
            json_info = json.load(json_reader)
        for index,an in enumerate(json_info):
            if index in self.black_list[type]:
                continue
            yolos= []
            gts=[]
            dts=[]
            scores = []
            img_name = an['img_name']
            img_name = self.__coco_name(img_name) if type == 'coco' else img_name
            self.img_list.append(img_name)
            pair = an['pair']
            for p in pair:
                scores.append(p['score'])
                yolo_bbox = np.asarray(p['yolo_bbox'])
                gt = p['gt']
                dt = p['dt']
                yolos.append(yolo_bbox)
                gts.append(gt)
                dts.append(dt)
            self.score_list.append(scores)
            self.yolo_list.append(yolos)
            self.gts_list.append(gts)
            self.dts_list.append(dts)
            self.data_type.append(type)
    def _load_annotations(self,annotations):
        for annotation in annotations:
            self.__load_annotations(annotation['name'],annotation['type'])


    def __len__(self):
        return len(self.img_list)


@DATASETS.register_module
class PoseDatatest(Dataset):
    def __init__(self, json_file, pipeline,img_dir):
        super(PoseDatatest, self).__init__()
        self.yolo_list = []

        self.img_list = []
        self.keypoints_list = []
        self.score_list = []
        self.pipeline = Compose(pipeline)
        self.img_dir  = img_dir[json_file['type']]
        self._load_annotations(json_file['name'])

    def _load_annotations(self,json_file):
        with open(json_file, 'r') as reader:
            json_reader = json.load(reader)

        for an in json_reader:
            self.yolo_list.append(an['bbox'])
            self.img_list.append(an['image_id'])
            self.keypoints_list.append(an['keypoints'])
            self.score_list.append(an['score'])

    def __getitem__(self, item):
        img_name = self.img_list[item]
        orig_img = os.path.join(self.img_dir, img_name)
        orig_img = [cv2.imread(orig_img)]
        scores = torch.from_numpy(np.asarray(self.score_list[item])).float().view(-1, 1)

        yolos = torch.from_numpy(np.asarray(self.yolo_list[item])).float().view(-1, 4)

        inps = torch.zeros(yolos.size(0), 3, opt.inputResH, opt.inputResW)
        pt1 = torch.zeros(yolos.size(0), 2)
        pt2 = torch.zeros(yolos.size(0), 2)
        assert self.pipeline is not None
        result = dict(orig_img = orig_img[0], im_name=img_name, boxes=yolos,scores=scores, inps=inps, pt1=pt1, pt2=pt2)
        result = self.pipeline(result)
        dts = self.keypoints_list[item]

        result['dts'] = dts
        result['item'] = item
        result['gts']  =None
        return result

    def __len__(self):
        return len(self.yolo_list)


class pose_generator():
    def __init__(self,model,device):
        self.model = model
        self.model.eval()
        if device =="cuda":
            self.model.cuda()
    def __call__(self,inps, orig_img, im_name, boxes, scores, pt1, pt2,gts_list,dts_list):
        if boxes is None or boxes.nelement() == 0:
            return None
        inps = inps.cuda()
        with torch.no_grad():
            hm, ret_features = self.model(inps)
        gts_epoch =[]
        dts_epoch = []
        h_4,w_4 = opt.outputResH,opt.outputResW
        if gts_list is None:
            gts_epoch = None
            dts_epoch = np.asarray(dts_list).reshape(-1,12,3)
            pre_keypoints = dts_epoch[:, ..., :2]
            hm_1_4 = transformBox_batch(pre_keypoints, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH,
                                        opt.outputResW)
            self.hm_normalize(hm_1_4,h_4,w_4)
            self.normalize_only(dts_epoch,pt1,pt2)
            dts = torch.from_numpy(dts_epoch).float()
            return dts,hm_1_4,ret_features
        else:
            for gts,dts in zip(gts_list,dts_list):
                gts = np.asarray(gts)
                gts_epoch.append(gts)
                dts = np.asarray(dts)
                dts_epoch.append(dts)
            gts_epoch = np.concatenate(gts_epoch,axis=0).astype(np.float32).copy()
            dts_epoch = np.concatenate(dts_epoch,axis=0).copy()

            # extract_feature_from_here
            pre_keypoints = dts_epoch[:, ..., :2]
            hm_1_4 = transformBox_batch(pre_keypoints, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH,
                                        opt.outputResW)
            self.hm_normalize(hm_1_4,h_4,w_4)
            # make it to -1,1
            self.normalize(dts_epoch,gts_epoch,pt1,pt2)
            dts = torch.from_numpy(dts_epoch).float()
            gts = torch.from_numpy(gts_epoch).float()

            return dts,gts,hm_1_4,ret_features


    def extract_features_joints(self,ret_features,hms):
        '''
        extract features from joint feature_map

        :return:
        '''

        joint_features = []


        for feature, hm_pred in zip(ret_features, hms):
            joint_feature = torch.zeros([feature.shape[0], feature.shape[1], hm_pred.shape[1]])
            for bz in range(feature.shape[0]):
                for joint in range(hm_pred.shape[1]):
                    joint_feature[bz, :, joint] = feature[bz, :, hm_pred[bz, joint, 1], hm_pred[bz, joint, 0]]
            joint_features.append(joint_feature)
        return joint_features
    def normalize_only(self,dts,pt1,pt2):
        num_joints = dts.shape[1]

        dts[:, :, 0] = dts[:, :, 0] - pt1[:, 0].unsqueeze(-1).repeat(1, num_joints).numpy()
        dts[:, :, 1] = dts[:, :, 1] - pt1[:, 1].unsqueeze(-1).repeat(1, num_joints).numpy()
        for bz in range(dts.shape[0]):
            x0,y0 = pt1[bz].numpy().tolist()
            x1,y1 = pt2[bz].numpy().tolist()
            w,h = x1-x0,y1-y0
            dts[bz,:,:2] = self.normalize_screen_coordinates(dts[bz,:,:2],w,h)
        return dts
    def inverse_normalize_only(self,dts,pt1,pt2):
        num_joints = dts.shape[1]

        for bz in range(dts.shape[0]):
            x0,y0 = pt1[bz].numpy().tolist()
            x1,y1 = pt2[bz].numpy().tolist()
            w,h = x1-x0,y1-y0
            dts[bz, :, :2] = self.inverse_normalize(dts[bz, :, :2], w, h)
        dts[:, :, 0] = dts[:, :, 0] + pt1[:, 0].unsqueeze(-1).repeat(1, num_joints).numpy()
        dts[:, :, 1] = dts[:, :, 1] + pt1[:, 1].unsqueeze(-1).repeat(1, num_joints).numpy()


    def normalize(self,dts,gts,pt1,pt2):

        num_joints = gts.shape[1]

        dts[:, :, 0] = dts[:, :, 0] - pt1[:, 0].unsqueeze(-1).repeat(1, num_joints).numpy()
        dts[:, :, 1] = dts[:, :, 1] - pt1[:, 1].unsqueeze(-1).repeat(1, num_joints).numpy()

        gts[:, :, 0] = gts[:, :, 0] - pt1[:, 0].unsqueeze(-1).repeat(1, num_joints).numpy()
        gts[:, :, 1] = gts[:, :, 1] - pt1[:, 1].unsqueeze(-1).repeat(1, num_joints).numpy()


        for bz in range(dts.shape[0]):
            x0,y0 = pt1[bz].numpy().tolist()
            x1,y1 = pt2[bz].numpy().tolist()
            w,h = x1-x0,y1-y0
            dts[bz,:,:2] = self.normalize_screen_coordinates(dts[bz,:,:2],w,h)
            gts[bz,:,:2] = self.normalize_screen_coordinates(gts[bz,:,:2],w,h)



    def normalize_screen_coordinates(self,X,w,h):
        assert X.shape[-1] == 2
        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
        #Normalize

        X[:,0] = X[:,0] / float(w) - 0.5
        X[:,1] = X[:,1] / float(h) - 0.5
        return X*2
    def inverse_normalize(self,Y, w, h):
        assert Y.shape[-1] == 2

        Y/=2.
        Y+=0.5
        Y[:,0] = Y[:,0]*float(w)
        Y[:,1] = Y[:,1]*float(h)
        return Y
    def hm_normalize(self,x,h,w):
        x[:,:,0] /=w
        x[:,:,1] /=h
        x-=0.5
        x*=2



def transfer(orig_img, im_name, boxes, scores, inps, pt1, pt2):
    #return inps,orig_img,im_name,boxes,scores,pt1,pt2
    if boxes is None or boxes.nelement() == 0:
        return (None, orig_img, im_name, boxes, scores, None, None)
    inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    # to refine our yolo results

    # to crop the corresponding inp
    # in here we only scale 0.3 for our detection bbox
    # crop_from_dets
    inps, pt1, pt2 = crop_from_dets_train_single(inp, boxes, inps, pt1, pt2)
    return (inps, orig_img, im_name, boxes, scores, pt1, pt2)

def crop_from_dets_train_single(img, boxes, inps, pt1, pt2):
    '''
    Crop human from origin image according to Dectecion Results
    '''

    tmp_img = img

    #to subtract mean RGB 0.406 0.457 0.480
    tmp_img[0].add_(-0.406)
    tmp_img[1].add_(-0.457)
    tmp_img[2].add_(-0.480)
    for i, box in enumerate(boxes):
        upLeft = torch.Tensor(
            (float(box[0]), float(box[1])))
        bottomRight = torch.Tensor(
            (float(box[2]), float(box[3])))
        try:
            inps[i] = cropBox(tmp_img.clone(), upLeft, bottomRight, opt.inputResH, opt.inputResW)
        except IndexError:
            print(tmp_img.shape)
            print(upLeft)
            print(bottomRight)
            print('===')
        pt1[i] = upLeft
        pt2[i] = bottomRight
    return inps, pt1, pt2





# using to build edge using GCN
def build_adj_mx_from_edges(num_joints,edge=EDGE):

    return adj_mx_from_edges(num_joints,edge,False)



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
