from .crowdposetools.coco import COCO
from .crowdposetools.cocoeval import COCOeval

def eval_results(preds,target_json):
    gt_file = target_json
    cocoGt = COCO(gt_file)
    cocoDt = cocoGt.loadRes(preds)
    cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
    cocoEval.evaluate()
    cocoEval.accumulate()
    mAP,AP50,AP75 = cocoEval.summarize()
    return mAP,AP50,AP75

