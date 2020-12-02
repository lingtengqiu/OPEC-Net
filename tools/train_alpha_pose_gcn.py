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
import os
from torch.utils.data import DataLoader
import torch.nn as nn
from engineer.datasets.loader.build_loader import train_loader_collate_fn,test_loader_collate_fn
from engineer.datasets.builder import build_dataset
from engineer.models.builder import build_generator,build_backbone
from engineer.core.train import  train_epochs
from utils import group_weight





if __name__ == "__main__":

    args = opt
    assert args.config is not None,"you must give your model config"
    cfg = Config.fromfile(args.config)


    # train_data_set = TrainSingerDataset(cfg.data.json_file, transfer=transfer,img_dir = cfg.data.img_dir,black_list=cfg.data.black_list)
    train_data_set = build_dataset(cfg.data.train)
    train_loader = DataLoader(train_data_set,batch_size=opt.trainBatch,shuffle=True,num_workers=4,collate_fn=train_loader_collate_fn)


    test_data_set = build_dataset(cfg.data.test)
    test_loader = DataLoader(test_data_set,batch_size=opt.validBatch,shuffle=False,num_workers=4,collate_fn=test_loader_collate_fn)

    cfg.checkpoints = os.path.join(cfg.checkpoints,cfg.name)
    if not os.path.exists(cfg.checkpoints):
        os.mkdir(cfg.checkpoints)


    if "Alpha.py" in args.config:
        pose_generator =None
    else:
        pose_generator = build_generator(cfg.pose_generator)
    device = torch.device("cuda")
    #gcn model maker

    model_pos = build_backbone(cfg.model).to(device).to(device)
    model_pos.train()

    #optim_machine
    param_list = []
    for module in model_pos.gcn_head:
        params_list = group_weight.group_weight(param_list,module,args.LR)
    for module in model_pos.heat_map_head:
        params_list = group_weight.group_weight(param_list,module,1e-3)
    for module in model_pos.generator_map:
        params_list = group_weight.group_weight(param_list,module,1e-3)

    criterion = nn.L1Loss(size_average=True,reduce=True).to(device)

    optimizer = torch.optim.Adam(params_list, lr=args.LR)
    cfg.LR=args.LR
    cfg.nEpochs = args.nEpochs

    # Init data writer

    train_epochs(model_pos, optimizer, cfg, args, train_loader, pose_generator, criterion,test_loader,cfg.pred_json)


