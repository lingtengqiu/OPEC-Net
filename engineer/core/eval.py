import  numpy as np
from collections import defaultdict
import json
from tqdm import tqdm
import cv2
import os
import torch
import copy
crowd_pose_dir = "../crowdpose/images"
from utils.metrics import eval_results

sigmas = np.array(
    [.79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .79, .79]) / 10.0
sigmas = sigmas[:12]
vars = (sigmas * 2) ** 2
bone = ([0,13],[1,13],[0,2],[2,4],[1,3],[3,5],[0,6],[1,7],[6,7],[6,8],[8,10],[7,9],[9,11],[12,13])
lr_stone = [15,20]


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0
def compute_oks_match(dts,gts,areas):


    if len(gts) == 0 or len(dts) == 0:
        return None
    ious = np.zeros((len(dts), len(gts)))


    # compute oks between each detection and ground truth object
    for j, gt in enumerate(gts):
        # create bounds for ignore regions(double the gt bbox)
        g = gt
        xg = g[:12,0];
        yg = g[:12,1];
        vg = g[:12,2]
        k1 = np.count_nonzero(vg > 0)

        for i, dt in enumerate(dts):
            d = dt
            xd = d[:,0];
            yd = d[:,1];

            if k1 > 0:
                # measure the per-keypoint distance if keypoints visible
                dx = xd - xg
                dy = yd - yg
            else:
                # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                return None
            e = (dx ** 2 + dy ** 2) / vars / (areas[j] + np.spacing(1)) / 2
            if k1 > 0:
                e = e[vg > 0]
            ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
    return ious
def get_gt(target_json):
    with open(target_json,'r') as reader:
        gt_json = json.load(reader)
    annotations = gt_json['annotations']
    id2gt =defaultdict(list)
    id2area = defaultdict(list)
    for an in annotations:
        image_id = an['image_id']
        image_name = "{}.jpg".format(image_id)

        area = an['bbox'][3]*an['bbox'][2]

        id2area[image_name].append(area)
        id2gt[image_name].append(np.asarray(an['keypoints']).reshape(-1,3))
    return id2gt,id2area
def vis_keypoints(best_pred_json,target_json):


    if isinstance(best_pred_json,str):
        with open(best_pred_json,'r') as reader:
            ans =json.load(reader)
    else:
        ans = best_pred_json


    id2gt,id2area = get_gt(target_json)



    vis_sets=defaultdict(list)

    ori_match = defaultdict(dict)
    for an in ans:
        img_name = an['image_id']
        keypoints = np.asarray(an['keypoints']).reshape(-1,3)
        vis_sets[img_name].append(keypoints)
    for key in tqdm(id2gt.keys()):
        gt = id2gt[key]
        dt = vis_sets[key]
        area = id2area[key]
        ious = compute_oks_match(dt,gt,area)
        if ious is None:
            continue
        gtm = np.zeros(len(gt))
        gtm[:]=-1
        dtm =np.zeros(len(dt))
        for gind in range(ious.shape[1]):
            # information about best match so far (m=-1 -> unmatched)
            iou = 0.3
            m = -1
            for dind in range(ious.shape[0]):
                # if this gt already matched, and not a crowd, continue
                if dtm[dind] > 0:
                    continue
                # continue to next gt unless better match made
                if ious[dind, gind] < iou:
                    continue
                # if match successful and best so far, store appropriately
                iou = ious[dind, gind]
                m = dind
            # if match made store id of match for both dt and gt
            if m == -1:
                continue
            dtm[m] = gind+1
            gtm[gind] = m
        gtm = gtm.astype(np.int).tolist()
        _keypoints = []
        _ious =[]
        _gt = []
        for gind,dind in enumerate(gtm):
            if dind<0:
                continue
            keypoint = dt[dind]
            gt_keypoint = gt[gind]
            keypoint[:,2] = gt_keypoint[:12,2]
            keypoint = keypoint.astype(np.int).tolist()
            _keypoints.append(keypoint)
            _gt.append(gt[gind])
            _ious.append(ious[dind,gind])
        ori_match[key]['keypoint'] = _keypoints
        ori_match[key]['iou'] = _ious
        ori_match[key]['gt'] = _gt
    return ori_match
def compare_vis(pred,ori):
    for key in pred.keys():
        pred_match = pred[key]
        ori_match = ori[key]
        pred_sum = sum(pred_match['iou'])
        ori_sum = sum(ori_match['iou'])
        if(pred_sum-ori_sum) >0 and (pred_sum-ori_sum)/len(ori_match['iou'])>0.03:
            pred_keypoints = pred_match['keypoint']
            ori_keypoints = ori_match['keypoint']
            gt_keypoints = ori_match['gt']
            if len(pred_keypoints) != len(ori_keypoints):
                continue
            colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (220, 20, 60), (255, 165, 0), (0, 191, 255), (46, 139, 87)]
            img_ori = cv2.imread(os.path.join(crowd_pose_dir,key))
            img_pred = img_ori.copy()
            for ind,(pred_keypoint,ori_keypoint,gt_keypoint) in enumerate(zip(pred_keypoints,ori_keypoints,gt_keypoints)):
                gt_keypoint = np.asarray(gt_keypoint)
                gt_keypoint.astype(np.int).reshape(-1,3).tolist()
                pred_keypoint.append(gt_keypoint[12])
                pred_keypoint.append(gt_keypoint[13])
                ori_keypoint.append(gt_keypoint[12])
                ori_keypoint.append(gt_keypoint[13])


                for b in bone:
                    if pred_keypoint[b[0]][2] == 0 or pred_keypoint[b[1]][2] == 0:
                        continue
                    img_pred = cv2.line(img_pred,tuple(pred_keypoint[b[0]][:2]),tuple(pred_keypoint[b[1]][:2]),colors[ind%7],3)
                    img_ori = cv2.line(img_ori,tuple(ori_keypoint[b[0]][:2]),tuple(ori_keypoint[b[1]][:2]),colors[ind%7],3)
            new_img = np.concatenate((img_ori,img_pred),axis=1)
            cv2.imwrite(os.path.join("vis_crowd","{}_{}.jpg".format(key.replace(".jpg",""),pred_sum-ori_sum)),new_img)


def eval_map(alpha_pose_generator,model_pose,test_dataloader,pred_json,best_json,target_json):

    id2bbox =defaultdict(list)
    id2keypoints = defaultdict(list)
    id2scores = defaultdict(list)
    id2cat = defaultdict(list)
    with open(best_json,'r') as reader:
        best_match =json.load(reader)
    for an in best_match:
        id2bbox[an['image_id']].append(an['bbox'])
        id2keypoints[an['image_id']].append(an['keypoints'])
        id2scores[an['image_id']].append(an['score'])
        id2cat[an['image_id']].append(an['category_id'])

    # ori_results_vis = vis_keypoints(best_json)


    print("eval the mAP")
    model_pose.eval()
    torch.set_grad_enabled(False)
    #json_file in here
    with open(pred_json) as reader:
        json_file_ori = json.load(reader)
        json_file_0_03 = copy.deepcopy(json_file_ori)

    cnt = 1
    for batches in tqdm(test_dataloader):
        cnt+=1
        inps,orig_img,img_name,boxes,scores,pt1,pt2,gts,dts,item = batches

        best_pose_match =[]
        for name,box in zip(img_name,boxes):
            best_box = id2bbox[name]
            best_box = np.asarray(best_box)
            box = box.cpu().numpy()
            ious = []
            for b_box in best_box:
                ious.append(compute_iou(b_box,box))
            ious = np.asarray(ious)
            ind = np.argmax(ious)
            if ious[ind]>0.9:
                best_pose_match.append(ind)
            else:
                best_pose_match.append(None)


        dts, hm_4,ret_features = alpha_pose_generator(inps, orig_img, img_name, boxes, scores,pt1, pt2, gts, dts)

        dts = dts.cuda()
        hm_4 = hm_4.cuda()
        with torch.no_grad():
            out_2d,heat_map_regress,inter_gral_x = model_pose(dts,hm_4,ret_features)
            out_2d = out_2d[2].cpu().detach().numpy()
            labels = dts[:,...,2:].repeat(1,1,3).cpu().detach().numpy()
            dts = dts.cpu().detach().numpy()
            inter_gral_x = inter_gral_x.cpu().detach().numpy()
            scores = dts[:,...,2:]
            alpha_pose_generator.inverse_normalize_only(out_2d,pt1,pt2)
            alpha_pose_generator.inverse_normalize_only(dts, pt1, pt2)
            alpha_pose_generator.inverse_normalize_only(inter_gral_x[...,:2], pt1, pt2)
            dts_003 = dts.copy()
            adj_joints = np.concatenate([out_2d,scores],axis=-1)
            dts_003[labels<0.2] = adj_joints[labels<0.2]
            for bz in range(dts_003.shape[0]):
                index = item[bz]
                name = img_name[bz]
                ind = best_pose_match[bz]
                if ind is not None:
                    id2keypoints[name][ind] = dts_003[bz,...].reshape(-1).tolist()
                json_file_ori[index]['keypoints'] = adj_joints[bz,...].reshape(-1).tolist()
                json_file_0_03[index]['keypoints'] = dts_003[bz,...].reshape(-1).tolist()
    new_best_match =[]
    for key,keypoints in id2keypoints.items():
        for keypoint,box,sco,cat in zip(keypoints,id2bbox[key],id2scores[key],id2cat[key]):
            ele = {}
            ele['keypoints'] = keypoint
            ele['bbox'] = box
            ele['score'] = sco
            ele['image_id'] = key
            ele['category_id'] = cat
            new_best_match.append(ele)



    #test results on best_best_match method in here and only PIN method
    best_match_map,_,_ = eval_results(new_best_match,target_json)

    ap003,_,_ = eval_results(json_file_0_03,target_json)
    return ap003,best_match_map