'''
@author: lingteng qiu
@version:1.0
'''

import torch
def train_loader_collate_fn(batches):
    inps_list = []
    orig_img_list= []
    img_name_list= []
    boxes_list= []
    scores_list= []
    pt1_list= []
    pt2_list =[]
    gts_list =[]
    dts_list = []
    for i in range(len(batches)):
        result = batches[i]
        inps = result['inps']
        orig_img = result['orig_img']
        img_name = result['im_name']
        boxes = result['boxes']
        scores = result['scores']
        pt1 = result['pt1']
        pt2 = result['pt2']
        gts = result['gts']
        dts = result['dts']

        inps_list.append(inps)
        orig_img_list.append(orig_img)
        img_name_list.append(img_name)
        boxes_list.append(boxes)
        scores_list.append(scores)
        pt1_list.append(pt1)
        pt2_list.append(pt2)
        gts_list.append(gts)
        dts_list.append(dts)
    inps = torch.cat(inps_list,dim=0)
    boxes = torch.cat(boxes_list,dim=0)
    scores = torch.cat(scores_list,dim=0)
    pt1 = torch.cat(pt1_list,dim=0)
    pt2 = torch.cat(pt2_list,dim=0)

    return inps,orig_img_list,img_name_list,boxes,scores,pt1,pt2,gts_list,dts_list

# One method to use in eval progress
def test_loader_collate_fn(batches):
    inps_list = []
    orig_img_list= []
    img_name_list= []
    boxes_list= []
    scores_list= []
    pt1_list= []
    pt2_list =[]
    dts_list = []
    item_list = []
    for i in range(len(batches)):
        result = batches[i]
        inps = result['inps']
        orig_img = result['orig_img']
        img_name = result['im_name']
        boxes = result['boxes']
        scores = result['scores']
        pt1 = result['pt1']
        pt2 = result['pt2']
        gts = result['gts']
        dts = result['dts']
        item = result['item']
        inps_list.append(inps)
        orig_img_list.append(orig_img)
        img_name_list.append(img_name)
        boxes_list.append(boxes)
        scores_list.append(scores)
        pt1_list.append(pt1)
        pt2_list.append(pt2)
        dts_list.append(dts)
        item_list.append(item)
    inps = torch.cat(inps_list,dim=0)
    boxes = torch.cat(boxes_list,dim=0)
    scores = torch.cat(scores_list,dim=0)
    pt1 = torch.cat(pt1_list,dim=0)
    pt2 = torch.cat(pt2_list,dim=0)
    return inps,orig_img_list,img_name_list,boxes,scores,pt1,pt2,None,dts_list,item_list