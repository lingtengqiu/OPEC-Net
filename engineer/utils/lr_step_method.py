#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Learning rate policy."""

import math


def get_lr_at_epoch(cfg, cur_epoch):
    """
    Retrieve the learning rate of the current epoch with the option to perform
    warm up in the beginning of the training stage.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    lr = get_lr_func(cfg.lr_policy)(cfg, cur_epoch)
    # Perform warm up.
    if cur_epoch < cfg.warm_epoch:
        lr_start = cfg.lr_warm_up
        lr_end = get_lr_func(cfg.lr_policy)(
            cfg, cfg.warm_epoch
        )

        alpha = (lr_end - lr_start) / cfg.warm_epoch
        lr = cur_epoch * alpha + lr_start
    return lr


def lr_func_cosine(cfg, cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    cosine learning rate schedule. Details can be found in:
    Ilya Loshchilov, and  Frank Hutter
    SGDR: Stochastic Gradient Descent With Warm Restarts.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return max(
        cfg.LR
        * (math.cos(math.pi * cur_epoch / cfg.nEpochs) + 1.0)
        * 0.5,1e-5
    )


def lr_func_steps_with_relative_lrs(cfg, cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    steps with relative learning rate schedule.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    ind = get_step_index(cfg, cur_epoch)
    return cfg.SOLVER.LRS[ind] * cfg.SOLVER.BASE_LR


def get_step_index(cfg, cur_epoch):
    """
    Retrieves the lr step index for the given epoch.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    steps = cfg.SOLVER.STEPS + [cfg.SOLVER.MAX_EPOCH]
    for ind, step in enumerate(steps):  # NoQA
        if cur_epoch < step:
            break
    return ind - 1


def get_lr_func(lr_policy):
    """
    Given the configs, retrieve the specified lr policy function.
    Args:
        lr_policy (string): the learning rate policy to use for the job.
    """
    policy = "lr_func_" + lr_policy
    if policy not in globals():
        raise NotImplementedError("Unknown LR policy: {}".format(lr_policy))
    else:
        return globals()[policy]


def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decays.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return get_lr_at_epoch(cfg, cur_epoch)


def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr