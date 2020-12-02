CUDA_VISIBLE_DEVICES=0 python ./tools/test_alpha_pose_gcn.py --indir ../crowdpose/images/  --load_dirs $1 --validBatch 60  --dataset 'coco' --config $2
