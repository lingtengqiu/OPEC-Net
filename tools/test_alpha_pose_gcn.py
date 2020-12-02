'''
@author:lingteng qiu
@name:OPEC_GCN
'''
import sys
sys.path.append("./")
from opt import opt
from mmcv import Config
from engineer.SPPE.src.main_fast_inference import *
try:
    from utils.img import transformBox_batch
except ImportError:
    from engineer.SPPE.src.utils.img import transformBox_batch
from torch.utils.data import DataLoader
import torch.nn as nn
from engineer.datasets.loader.build_loader import train_loader_collate_fn,test_loader_collate_fn
from engineer.datasets.builder import build_dataset
from engineer.models.builder import build_generator,build_backbone
from engineer.core.eval import eval_map
import os





if __name__ == "__main__":

    args = opt
    assert args.config is not None,"you must give your model config"
    cfg = Config.fromfile(args.config)

    checkpoints = os.path.join(args.load_dirs,"best_checkpoint.pth")

    assert os.path.exists(checkpoints)


    # train_data_set = TrainSingerDataset(cfg.data.json_file, transfer=transfer,img_dir = cfg.data.img_dir,black_list=cfg.data.black_list)
    train_data_set = build_dataset(cfg.data.train)
    train_loader = DataLoader(train_data_set,batch_size=opt.trainBatch,shuffle=True,num_workers=4,collate_fn=train_loader_collate_fn)


    test_data_set = build_dataset(cfg.data.test)
    test_loader = DataLoader(test_data_set,batch_size=opt.validBatch,shuffle=False,num_workers=4,collate_fn=test_loader_collate_fn)



    if "Alpha.py" in args.config:
        pose_generator =None
    else:
        pose_generator = build_generator(cfg.pose_generator)
    device = torch.device("cuda")
    #gcn model maker

    model_pos = build_backbone(cfg.model)
    model_pos.load_state_dict(torch.load(checkpoints))
    model_pos.to(device)
    model_pos.eval()

    criterion = nn.L1Loss(size_average=True,reduce=True).to(device)

    # Init data writer

    mAP, ap = eval_map(pose_generator, model_pos, test_loader, cfg.pred_json, best_json=cfg.best_json,
                       target_json=cfg.target_json)

    print(mAP,ap)


