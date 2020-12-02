'''
@author:lingteng qiu
@name:OPEC_GCN
'''
import os
from utils.bar import Bar
from utils.structure import AverageMeter
import time
import torch.nn as nn
import torch
from engineer.core.eval import eval_map
from engineer.utils import lr_step_method as optim

def train_epochs(model_pos,optimizer,cfg,args,train_loader,pose_generator,criterion,test_loader,pred_json):
    epochs = args.nEpochs

    best_map =None
    writer_map = open(os.path.join(cfg.checkpoints,"mAP.txt"),'w')
    for epoch in range(epochs):
        print("Epoch :{}".format(epoch))
        #average
        epoch_loss_2d_pos = AverageMeter()
        epoch_loss_heat_map = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()
        bar = Bar('Train', max=len(train_loader))
        for _,batches in enumerate(train_loader):
            inps, orig_img_list, img_name_list, boxes, scores, pt1, pt2, gts_list, dts_list = batches
            if pose_generator is not None:
                dts,gt_2d,hm_4,ret_features = pose_generator(inps,orig_img_list,img_name_list,boxes,scores,pt1,pt2,gts_list,dts_list)
                dts = dts.cuda()
                gt_2d = gt_2d.cuda()
                hm_4 = hm_4.cuda()
                ret_features = [ret.cuda() for ret in ret_features]
                bz = dts.shape[0]
                data_time.update(time.time() - end)
                out_2d, heat_map_regress, inter_gral_x = model_pos(dts, hm_4, ret_features)
                heat_map_regress = heat_map_regress.view(-1, 12, 2)
            else:
                out_2d, heat_map_regress, inter_gral_x, gt_2d, bz = model_pos(inps,orig_img_list,img_name_list,boxes,scores,pt1,pt2,gts_list,dts_list)
                heat_map_regress = heat_map_regress.view(-1, 12, 2)
                data_time.update(time.time() - end)
            lr = optim.get_epoch_lr(epoch + float(_) / len(train_loader), cfg)
            optim.set_lr(optimizer, lr)




            optimizer.zero_grad()
            labels = gt_2d[:,...,2]
            labels = labels[:,:,None].repeat(1,1,2)
            gt_2d = gt_2d[:, ..., :2]
            gt_2d = gt_2d[labels > 0].view(-1, 2)
            out_2d_0 = out_2d[0][labels > 0].view(-1, 2)
            out_2d_1 = out_2d[1][labels > 0].view(-1, 2)
            out_2d_2 = out_2d[2][labels > 0].view(-1, 2)
            loss_2d_pos_0 = criterion(out_2d_0, gt_2d)
            loss_2d_pos_1 = criterion(out_2d_1, gt_2d)
            loss_2d_pos_2 = criterion(out_2d_2, gt_2d)
            loss_heat_map = criterion(heat_map_regress[labels>0].view(-1,2), gt_2d)
            loss_2d_pos = 0.3*loss_2d_pos_0+0.5*loss_2d_pos_1+loss_2d_pos_2+loss_heat_map
            loss_2d_pos.backward()

            if True:
                nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
            optimizer.step()
            epoch_loss_2d_pos.update(loss_2d_pos.item(),bz)
            epoch_loss_heat_map.update(loss_heat_map.item(), bz)
            batch_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                         '| Loss: {loss: .4f}| Loss_heat:{heat: .4f}| LR:{LR: .6f}' \
                .format(batch=_ + 1, size=len(train_loader), data=data_time.val, bt=batch_time.avg,
                        ttl=bar.elapsed_td, eta=bar.eta_td, loss=epoch_loss_2d_pos.avg,heat=epoch_loss_heat_map.avg,LR=lr)
            bar.next()
        bar.finish()
        mAP,ap = eval_map(pose_generator,model_pos,test_loader,pred_json,best_json=cfg.best_json,target_json=cfg.target_json)
        writer_map.write("{}\t{}\t{}\n".format(epoch,mAP,ap))
        writer_map.flush()
        if best_map is None or best_map<mAP:
            best_map=mAP
            torch.save(model_pos.state_dict(), os.path.join(cfg.checkpoints, "best_checkpoint.pth"))

        model_pos.train()
        torch.set_grad_enabled(True)
        torch.save(model_pos.state_dict(),os.path.join(cfg.checkpoints,"{}.pth".format(epoch)))