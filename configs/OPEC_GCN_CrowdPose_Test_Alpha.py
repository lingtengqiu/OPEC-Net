import numpy as np
model = dict(
    type='Alpha_SemGCN_Attention',
    adj = ([0,1],[0,2],[2,4],[1,3],[3,5],[0,6],[1,7],[6,7],[6,8],[8,10],[7,9],[9,11]),
    num_joints = 12,
    hid_dim = [128,128,128,128,128],
    coords_dim=(3, 2),
    p_dropout=None,
    model="faster",
    outputResH=80,
    outputResW=64,
    inputResW=256,
    inputResH=320
)
pose_generator=dict(
    type = "Pose_Generator",
    model = "faster",
    outputResH = 80,
    outputResW = 64,
    inputResW = 256,
    inputResH = 320
)
dataset_type = 'CrowdPose'
train_pipeline = [
    dict(type='crop_large',RGB=[-0.406,-0.457,-0.480],inputResH = 320,inputResW = 256),
]

val_pipeline = [
    dict(type='crop_large',RGB=[-0.406,-0.457,-0.480],inputResH = 320,inputResW = 256),
]
data = dict(
    train=dict(
    type = "PoseDataset",
    # json_file = [dict(name = "./train_process_datasets/train_crowd_train.json",type = "crowd_pose")],
    json_file=[dict(name='train_process_datasets/coco_pose_45.json', type="coco"),\
                   dict(name="./train_process_datasets/train_crowd_train.json", type="crowd_pose")],
    img_dir={"crowd_pose": "../crowdpose/images", 'coco': "../train2017"},
    black_list ={
        "coco": [783, 859, 1100, 2173, 2196, 2468, 2473, 2878, 3064, 3304, 3477, 3488, 3687, 4190, 4959, 4963, 5382, 5574, 5937, 6124, 6210, 6312, 6386, 6458, 6631, 7044, 7147, 7487, 8115, 8811, 8865, 8867, 9081, 9585, 10017, 10261, 10262, 10410, 10601, 10775, 10779, 10831, 10960, 10969, 10997, 11125, 11210, 11288, 11349, 11897, 12167, 12317, 12340, 12413, 12632, 13561, 13668, 13843, 13931, 14038, 14316, 14318, 14376, 14377, 14495, 14598, 14708, 14933, 14979, 15058, 15170, 15391, 15708, 15747, 15788, 16011, 16012, 16143, 16391, 16402, 16777, 16844, 16858, 17198, 17301, 17558, 17576, 17955, 18181, 18486, 18619, 18636, 18726, 18881, 19165, 19221, 19942, 20724, 20743, 21108, 21428, 21834, 23008, 23432, 23543, 23696, 24108, 24475, 24837, 25491, 25707, 25993, 26059, 26109, 26175, 26208, 26241, 26307, 26392, 26878, 26957, 27310, 27373, 27462, 27883, 27975, 27986, 28090, 28130, 28306, 28593, 28616, 28934, 28970, 29090, 29266, 29858, 29954, 30905, 31379, 31446, 31488, 31527, 31593, 31835, 31876, 32033, 32166, 32202, 33062, 33425, 33467, 33767, 34004, 34014, 34114, 34261, 34379, 34693, 34969, 35107, 35381, 35449, 35815, 35923, 36719, 36935, 37275, 37449, 37462, 37618, 38253, 38387, 38445, 39164, 39303, 39528, 39733, 40859, 41020, 41040, 41058, 41447, 41937, 42091, 42110, 42386, 44042, 45083, 45560, 45758, 45772, 45875, 46219, 46356, 46450, 46986, 47106, 47733, 47878],
        "crowd_pose_test":[37,713,748,923,1647,2376,3003,3051,3187,3205,4224,4371,4492,4886,4945,5094,5253,5605,6198,6218,6574],
        "crowd_pose": [534, 553, 1782, 2481, 4127, 4379, 5812, 6268, 6712, 6764, 7483, 7792, 7979, 8028, 8824, 9203,9477]
    },
    pipeline=train_pipeline
    ),
    test=dict(
        type="PoseDatatest",
        json_file = dict(name="./test_process_datasets/test_compute_map++.json",type="crowd_pose"),
        pipeline = val_pipeline,
        img_dir={"crowd_pose": "../crowdpose/images", 'coco': "../train2017"},
    )
)
#save_info
checkpoints = "./checkpoints"
# optimizer
optim_para=dict(
    optimizer = dict(type='Adam',lr=0.0005),
    lr_decay=2,
    lr_gamma= 0.96
)

#init params
sigmas = np.array(
    [.79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .79, .79]) / 10.0
sigmas = sigmas[:12]
vars = (sigmas * 2) ** 2
bone = ([0,13],[1,13],[0,2],[2,4],[1,3],[3,5],[0,6],[1,7],[6,7],[6,8],[8,10],[7,9],[9,11],[12,13])
target_json = './test_process_datasets/crowdpose_test.json'
pred_json = "./test_process_datasets/test_compute_map++.json"
best_json ="./test_process_datasets/pred_test_best_match.json"
crowd_pose_dir = "../crowdpose/images"
lr_stone = [15,20]

#solver
lr_policy="cosine"
lr_warm_up = 1e-5
warm_epoch=1
LR=1e-3
nEpochs=25

name=__name__