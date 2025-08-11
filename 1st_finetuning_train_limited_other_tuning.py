import os
# GLIPへのパスを指定
os.chdir('./GLIP')

# ft_type = "full"
# ft_type = "prompt"
ft_type = "linear_prob"
shot_num = "all"

MAX_TRAIN_ANNOTATIONS_PER_CATEGORY = 400
MAX_VAL_ANNOTATIONS_PER_CATEGORY = MAX_TRAIN_ANNOTATIONS_PER_CATEGORY // 3  # trainの1/3

ft_tasks_file = f"../../bridge_data/configs/20250411共有_5cat_train_{MAX_TRAIN_ANNOTATIONS_PER_CATEGORY}_val_{MAX_VAL_ANNOTATIONS_PER_CATEGORY}_valid.yaml"
dir_name = f"20250411共有_5cat_train_{MAX_TRAIN_ANNOTATIONS_PER_CATEGORY}_val_{MAX_VAL_ANNOTATIONS_PER_CATEGORY}_valid"

output_path = "../"

seed_value =  42

# 必要なパッケージのimport（前準備）
import random
import numpy as np
import torch

import argparse
import os
import glob

import pdb
import torch
import requests
from maskrcnn_benchmark.config import cfg, try_to_find
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.engine.alter_trainer import do_train as alternative_train
from maskrcnn_benchmark.engine.stage_trainer import do_train as multi_stage_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, is_main_process
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.metric_logger import (MetricLogger, TensorboardLogger)
import shutil

import datetime
import logging
import sys
import os
import math
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size, all_gather, is_main_process, broadcast_data, get_rank
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.ema import ModelEma
from maskrcnn_benchmark.utils.amp import autocast, GradScaler
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
import pdb

from maskrcnn_benchmark.engine.trainer import *
from maskrcnn_benchmark.modeling.detector import *
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.rpn.vldyhead import *

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
torch_fix_seed(seed_value)

def removekey(d, prefix):
    r = dict(d)
    listofkeys = []
    for key in r.keys():
        if key.startswith(prefix):
            listofkeys.append(key)
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        r.pop(key)
    return r

def train(cfg, local_rank, distributed, zero_shot, skip_optimizer_resume=False, save_config_path = None):

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=0 #<TODO> Sample data from resume is disabled, due to the conflict with max_epoch
    )
    if cfg.TEST.DURING_TRAINING:
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
        data_loaders_val = data_loaders_val[0]
    else:
        data_loaders_val = None
    
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)


    if cfg.MODEL.LINEAR_PROB:
        assert cfg.MODEL.BACKBONE.FREEZE, "For linear probing, backbone should be frozen!"
        if hasattr(model.backbone, 'fpn'):
            assert cfg.MODEL.FPN.FREEZE, "For linear probing, FPN should be frozen!"
    if cfg.MODEL.BACKBONE.FREEZE:
        for p in model.backbone.body.parameters():
            p.requires_grad = False
    if cfg.MODEL.FPN.FREEZE:
        for p in model.backbone.fpn.parameters():
            p.requires_grad = False
    if cfg.MODEL.RPN.FREEZE:
        for p in model.rpn.parameters():
            p.requires_grad = False
    if cfg.MODEL.LINEAR_PROB:
        if model.rpn is not None:
            for key, p in model.rpn.named_parameters():
                if not ('bbox_pred' in key or 'cls_logits' in key or 'centerness' in key or 'cosine_scale' in key or 'dot_product_projection_text' in key or 'head.log_scale' in key or 'head.bias_lang' in key or 'head.bias0' in key):
                    p.requires_grad = False
        if model.roi_heads is not None:
            for key, p in model.roi_heads.named_parameters():
                if not ('bbox_pred' in key or 'cls_logits' in key or 'centerness' in key or 'cosine_scale' in key or 'dot_product_projection_text' in key or 'head.log_scale' in key or 'head.bias_lang' in key or 'head.bias0' in key):
                    p.requires_grad = False
    if cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER:
        if model.rpn is not None:
            for key, p in model.rpn.named_parameters():
                if 'tunable_linear' in key:
                    p.requires_grad = True

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
    
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=cfg.SOLVER.FIND_UNUSED_PARAMETERS
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    if checkpointer.has_checkpoint():
        extra_checkpoint_data = checkpointer.load(skip_optimizer=skip_optimizer_resume)
        arguments.update(extra_checkpoint_data)
    else:
        state_dict = checkpointer._load_file(try_to_find(cfg.MODEL.WEIGHT))
        checkpointer._load_model(state_dict)

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    meters = MetricLogger(delimiter="  ")

    if zero_shot:
        assert False
        #return model
    
    if is_main_process():
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(name, " : Not Frozen")
            else:
                print(name, " : Frozen")
    report_freeze_options(cfg)
    if cfg.DATASETS.ALTERNATIVE_TRAINING:
        alternative_train(
            model,
            data_loader,
            optimizer,
            scheduler,
            checkpointer,
            device,
            checkpoint_period,
            arguments,
        )
    elif cfg.DATASETS.MULTISTAGE_TRAINING:
        arguments['epoch_per_stage'] = cfg.SOLVER.MULTI_MAX_EPOCH
        multi_stage_train(
            model,
            data_loader,
            optimizer,
            scheduler,
            checkpointer,
            device,
            checkpoint_period,
            arguments,
        )
    else:
        meters = MetricLogger(delimiter="  ")
        do_train(
            cfg,
            model,
            data_loader,
            optimizer,
            scheduler,
            checkpointer,
            device,
            checkpoint_period,
            arguments,
            data_loaders_val,
            meters=meters
        )

    return model


def test(cfg, model, distributed, verbose=False):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    log_dir = cfg.OUTPUT_DIR
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    dataset_names = cfg.DATASETS.TEST
    if isinstance(dataset_names[0], (list, tuple)):
        dataset_names = [dataset for group in dataset_names for dataset in group]
    output_folders = [None] * len(dataset_names)
    if log_dir:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(log_dir, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=cfg.MODEL.RPN_ONLY and cfg.MODEL.RPN_ARCHITECTURE=="RPN",
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            cfg=cfg
        )
        synchronize()
    if verbose:
        with open(os.path.join(output_folder, "bbox.csv")) as f:
            print(f.read())

def tuning_highlevel_override(cfg,):
    if cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "full":
        cfg.MODEL.BACKBONE.FREEZE = False
        cfg.MODEL.FPN.FREEZE = False
        cfg.MODEL.RPN.FREEZE = False
        cfg.MODEL.LINEAR_PROB = False
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = False
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = False
    elif cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "linear_prob":
        cfg.MODEL.BACKBONE.FREEZE = True
        cfg.MODEL.FPN.FREEZE = True
        cfg.MODEL.RPN.FREEZE = False
        cfg.MODEL.LINEAR_PROB = True
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = False
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = True
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = True
        cfg.MODEL.DYHEAD.USE_CHECKPOINT = False # Disable checkpoint
    elif cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "language_prompt_v1":
        cfg.MODEL.BACKBONE.FREEZE = True
        cfg.MODEL.FPN.FREEZE = True
        cfg.MODEL.RPN.FREEZE = True
        cfg.MODEL.LINEAR_PROB = False
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = False
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = False
    elif cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "language_prompt_v2":
        cfg.MODEL.BACKBONE.FREEZE = True
        cfg.MODEL.FPN.FREEZE = True
        cfg.MODEL.RPN.FREEZE = True
        cfg.MODEL.LINEAR_PROB = False
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = True
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = True
    elif cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "language_prompt_v3":
        cfg.MODEL.BACKBONE.FREEZE = True
        cfg.MODEL.FPN.FREEZE = True
        cfg.MODEL.RPN.FREEZE = True
        cfg.MODEL.LINEAR_PROB = True # Turn on linear probe
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = False
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = False # Turn on language backbone
    elif cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "language_prompt_v4":
        cfg.MODEL.BACKBONE.FREEZE = True
        cfg.MODEL.FPN.FREEZE = True
        cfg.MODEL.RPN.FREEZE = True
        cfg.MODEL.LINEAR_PROB = True # Turn on linear probe
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = True
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = True # Turn off language backbone
    return cfg

def report_freeze_options(cfg):
    print("Backbone Freeze:", cfg.MODEL.BACKBONE.FREEZE)
    print("FPN Freeze:", cfg.MODEL.FPN.FREEZE)
    print("RPN Freeze:", cfg.MODEL.RPN.FREEZE)
    print("Linear Probe:", cfg.MODEL.LINEAR_PROB)
    print("Language Freeze:", cfg.MODEL.LANGUAGE_BACKBONE.FREEZE)
    print("Linear Layer (True Prmopt Tuning):", cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER)
    print("High Level Override:", cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE)


if ft_type == "full":
    if shot_num == "all":
        class args:
            config_file = "configs/pretrain/glip_Swin_L.yaml"
            # config_file="configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
            ft_tasks = ft_tasks_file #TODO
            local_rank = 0
            skip_train = False
            skip_test = True
            skip_optimizer_resume = False
            custom_shot_and_epoch_and_general_copy = "0_200_1" #TODO
            shuffle_seeds = None
            evaluate_only_best_on_test = True
            push_both_val_and_test = True
            use_prepared_data = False
            opts = [
                "MODEL.WEIGHT", "MODEL/glip_large_model.pth",
                # "MODEL.WEIGHT", "MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth",
                "SOLVER.USE_AMP", True,
                "TEST.DURING_TRAINING", True, 
                "TEST.IMS_PER_BATCH", 2 ,#2 ,
                "SOLVER.IMS_PER_BATCH", 2 ,#2,

                "SOLVER.WEIGHT_DECAY", 0.05, #full model tuning

                "TEST.EVAL_TASK", "detection",
                "DATASETS.TRAIN_DATASETNAME_SUFFIX", "_grounding",
                "MODEL.BACKBONE.FREEZE_CONV_BODY_AT", 2,
                "MODEL.DYHEAD.USE_CHECKPOINT", True,
                "SOLVER.FIND_UNUSED_PARAMETERS", False,
                "SOLVER.TEST_WITH_INFERENCE", True,
                "SOLVER.USE_AUTOSTEP", True,
                "DATASETS.USE_OVERRIDE_CATEGORY", True, 
                "SOLVER.SEED", 10,
                "DATASETS.SHUFFLE_SEED", 3, 
                "DATASETS.USE_CAPTION_PROMPT", True,
                "DATASETS.DISABLE_SHUFFLE", True,

                # "SOLVER.STEP_PATIENCE", 3, #few shot
                "SOLVER.STEP_PATIENCE", 2, #all shot

                "SOLVER.CHECKPOINT_PER_EPOCH", 1.0, 

                # "SOLVER.AUTO_TERMINATE_PATIENCE", 8, #few shot
                "SOLVER.AUTO_TERMINATE_PATIENCE", 4, #all shot

                "SOLVER.MODEL_EMA", 0.0,

                "SOLVER.TUNING_HIGHLEVEL_OVERRIDE", "full", #full model tuning

                # "SOLVER.MAX_NEG_PER_BATCH", 10.5
            ]
            
    else:
        class args:
            config_file = "configs/pretrain/glip_Swin_L.yaml"
            # config_file="configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
            ft_tasks = ft_tasks_file #TODO
            local_rank = 0
            skip_train = False
            skip_test = True
            skip_optimizer_resume = False
            custom_shot_and_epoch_and_general_copy = str(shot_num)+"_200_1" #TODO
            shuffle_seeds = None
            evaluate_only_best_on_test = True
            push_both_val_and_test = True
            use_prepared_data = False
            opts = [
                "MODEL.WEIGHT", "MODEL/glip_large_model.pth",
                # "MODEL.WEIGHT", "MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth",
                "SOLVER.USE_AMP", True,
                "TEST.DURING_TRAINING", True, 
                "TEST.IMS_PER_BATCH", 2 ,
                "SOLVER.IMS_PER_BATCH", 2,

                "SOLVER.WEIGHT_DECAY", 0.05, #full model tuning

                "TEST.EVAL_TASK", "detection",
                "DATASETS.TRAIN_DATASETNAME_SUFFIX", "_grounding",
                "MODEL.BACKBONE.FREEZE_CONV_BODY_AT", 2,
                "MODEL.DYHEAD.USE_CHECKPOINT", True,
                "SOLVER.FIND_UNUSED_PARAMETERS", False,
                "SOLVER.TEST_WITH_INFERENCE", True,
                "SOLVER.USE_AUTOSTEP", True,
                "DATASETS.USE_OVERRIDE_CATEGORY", True, 
                "SOLVER.SEED", 10,
                "DATASETS.SHUFFLE_SEED", 3, 
                "DATASETS.USE_CAPTION_PROMPT", True,
                "DATASETS.DISABLE_SHUFFLE", True,

                "SOLVER.STEP_PATIENCE", 3, #few shot
                # "SOLVER.STEP_PATIENCE", 2, #all shot

                "SOLVER.CHECKPOINT_PER_EPOCH", 1.0, 

                "SOLVER.AUTO_TERMINATE_PATIENCE", 8, #few shot
                # "SOLVER.AUTO_TERMINATE_PATIENCE", 4, #all shot

                "SOLVER.MODEL_EMA", 0.0,

                "SOLVER.TUNING_HIGHLEVEL_OVERRIDE", "full", #full model tuning

                # "SOLVER.MAX_NEG_PER_BATCH", 10.5
            ]
        
        
elif ft_type=="prompt":
    if shot_num == "all":
        class args:
            config_file = "configs/pretrain/glip_Swin_L.yaml"
            # config_file="configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
            ft_tasks = ft_tasks_file
            local_rank = 0
            skip_train = False
            skip_test = True
            skip_optimizer_resume = False
            custom_shot_and_epoch_and_general_copy = "0_200_1" #TODO
            shuffle_seeds = None
            evaluate_only_best_on_test = True
            push_both_val_and_test = True
            use_prepared_data = False
            opts = [
                "MODEL.WEIGHT", "MODEL/glip_large_model.pth",
                # "MODEL.WEIGHT", "MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth",
                "SOLVER.USE_AMP", True,
                "TEST.DURING_TRAINING", True, 
                "TEST.IMS_PER_BATCH", 2 ,
                "SOLVER.IMS_PER_BATCH", 2,

                "SOLVER.WEIGHT_DECAY", 0.25, #prompt tuning

                "TEST.EVAL_TASK", "detection",
                "DATASETS.TRAIN_DATASETNAME_SUFFIX", "_grounding",
                "MODEL.BACKBONE.FREEZE_CONV_BODY_AT", 2,
                "MODEL.DYHEAD.USE_CHECKPOINT", True,
                "SOLVER.FIND_UNUSED_PARAMETERS", False,
                "SOLVER.TEST_WITH_INFERENCE", True,
                "SOLVER.USE_AUTOSTEP", True,
                "DATASETS.USE_OVERRIDE_CATEGORY", True, 
                "SOLVER.SEED", 10,
                "DATASETS.SHUFFLE_SEED", 3, 
                "DATASETS.USE_CAPTION_PROMPT", True,
                "DATASETS.DISABLE_SHUFFLE", False, #1027

                # "SOLVER.STEP_PATIENCE", 3, #few shot
                "SOLVER.STEP_PATIENCE", 2, #all shot

                "SOLVER.CHECKPOINT_PER_EPOCH", 1.0, 

                # "SOLVER.AUTO_TERMINATE_PATIENCE", 8, #few shot
                "SOLVER.AUTO_TERMINATE_PATIENCE", 4, #all shot

                "SOLVER.MODEL_EMA", 0.0,

                "SOLVER.TUNING_HIGHLEVEL_OVERRIDE", "language_prompt_v2", #prompt tuning

                "SOLVER.BASE_LR", 0.05, #prompt tuning

                # "SOLVER.MAX_NEG_PER_BATCH", 10.5
            ]
            
    else:
        class args:
            config_file = "configs/pretrain/glip_Swin_L.yaml"
            # config_file="configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
            ft_tasks = ft_tasks_file
            local_rank = 0
            skip_train = False
            skip_test = True
            skip_optimizer_resume = False
            custom_shot_and_epoch_and_general_copy = str(shot_num)+"_200_1" #TODO
            shuffle_seeds = None
            evaluate_only_best_on_test = True
            push_both_val_and_test = True
            use_prepared_data = False
            opts = [
                "MODEL.WEIGHT", "MODEL/glip_large_model.pth",
                # "MODEL.WEIGHT", "MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth",
                "SOLVER.USE_AMP", True,
                "TEST.DURING_TRAINING", True, 
                "TEST.IMS_PER_BATCH", 2 ,
                "SOLVER.IMS_PER_BATCH", 2,

                "SOLVER.WEIGHT_DECAY", 0.25, #prompt tuning

                "TEST.EVAL_TASK", "detection",
                "DATASETS.TRAIN_DATASETNAME_SUFFIX", "_grounding",
                "MODEL.BACKBONE.FREEZE_CONV_BODY_AT", 2,
                "MODEL.DYHEAD.USE_CHECKPOINT", True,
                "SOLVER.FIND_UNUSED_PARAMETERS", False,
                "SOLVER.TEST_WITH_INFERENCE", True,
                "SOLVER.USE_AUTOSTEP", True,
                "DATASETS.USE_OVERRIDE_CATEGORY", True, 
                "SOLVER.SEED", 10,
                "DATASETS.SHUFFLE_SEED", 3, 
                "DATASETS.USE_CAPTION_PROMPT", True,
                "DATASETS.DISABLE_SHUFFLE", True,

                "SOLVER.STEP_PATIENCE", 3, #few shot
                # "SOLVER.STEP_PATIENCE", 2, #all shot

                "SOLVER.CHECKPOINT_PER_EPOCH", 1.0, 

                "SOLVER.AUTO_TERMINATE_PATIENCE", 8, #few shot
                # "SOLVER.AUTO_TERMINATE_PATIENCE", 4, #all shot

                "SOLVER.MODEL_EMA", 0.0,

                "SOLVER.TUNING_HIGHLEVEL_OVERRIDE", "language_prompt_v2", #prompt tuning

                "SOLVER.BASE_LR", 0.05, #prompt tuning

                # "SOLVER.MAX_NEG_PER_BATCH", 10.5
            ]

elif ft_type == "linear_prob":
    if shot_num == "all":
        class args:
            config_file = "configs/pretrain/glip_Swin_L.yaml"
            # config_file="configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
            ft_tasks = ft_tasks_file
            local_rank = 0
            skip_train = False
            skip_test = True
            skip_optimizer_resume = False
            custom_shot_and_epoch_and_general_copy = "0_200_1" #TODO
            shuffle_seeds = None
            evaluate_only_best_on_test = True
            push_both_val_and_test = True
            use_prepared_data = False
            opts = [
                "MODEL.WEIGHT", "MODEL/glip_large_model.pth",
                # "MODEL.WEIGHT", "MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth",
                "SOLVER.USE_AMP", True,
                "TEST.DURING_TRAINING", True, 
                "TEST.IMS_PER_BATCH", 2 ,
                "SOLVER.IMS_PER_BATCH", 2,

                "TEST.EVAL_TASK", "detection",
                "DATASETS.TRAIN_DATASETNAME_SUFFIX", "_grounding",
                "MODEL.BACKBONE.FREEZE_CONV_BODY_AT", 2,
                "MODEL.DYHEAD.USE_CHECKPOINT", True,
                "SOLVER.FIND_UNUSED_PARAMETERS", False,
                "SOLVER.TEST_WITH_INFERENCE", True,
                "SOLVER.USE_AUTOSTEP", True,
                "DATASETS.USE_OVERRIDE_CATEGORY", True, 
                "SOLVER.SEED", 10,
                "DATASETS.SHUFFLE_SEED", 3, 
                "DATASETS.USE_CAPTION_PROMPT", True,
                "DATASETS.DISABLE_SHUFFLE", False, #1027

                # "SOLVER.STEP_PATIENCE", 3, #few shot
                "SOLVER.STEP_PATIENCE", 2, #all shot

                "SOLVER.CHECKPOINT_PER_EPOCH", 1.0, 

                # "SOLVER.AUTO_TERMINATE_PATIENCE", 8, #few shot
                "SOLVER.AUTO_TERMINATE_PATIENCE", 4, #all shot

                "SOLVER.MODEL_EMA", 0.0,

                "SOLVER.TUNING_HIGHLEVEL_OVERRIDE", "linear_prob",

            ]
    else:
        class args:
            config_file = "configs/pretrain/glip_Swin_L.yaml"
            # config_file="configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
            ft_tasks = ft_tasks_file
            local_rank = 0
            skip_train = False
            skip_test = True
            skip_optimizer_resume = False
            custom_shot_and_epoch_and_general_copy = str(shot_num)+"_200_1" #TODO
            shuffle_seeds = None
            evaluate_only_best_on_test = True
            push_both_val_and_test = True
            use_prepared_data = False
            opts = [
                "MODEL.WEIGHT", "MODEL/glip_large_model.pth",
                # "MODEL.WEIGHT", "MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth",
                "SOLVER.USE_AMP", True,
                "TEST.DURING_TRAINING", True, 
                "TEST.IMS_PER_BATCH", 2 ,
                "SOLVER.IMS_PER_BATCH", 2,

                "TEST.EVAL_TASK", "detection",
                "DATASETS.TRAIN_DATASETNAME_SUFFIX", "_grounding",
                "MODEL.BACKBONE.FREEZE_CONV_BODY_AT", 2,
                "MODEL.DYHEAD.USE_CHECKPOINT", True,
                "SOLVER.FIND_UNUSED_PARAMETERS", False,
                "SOLVER.TEST_WITH_INFERENCE", True,
                "SOLVER.USE_AUTOSTEP", True,
                "DATASETS.USE_OVERRIDE_CATEGORY", True, 
                "SOLVER.SEED", 10,
                "DATASETS.SHUFFLE_SEED", 3, 
                "DATASETS.USE_CAPTION_PROMPT", True,
                "DATASETS.DISABLE_SHUFFLE", True,

                "SOLVER.STEP_PATIENCE", 3, #few shot
                # "SOLVER.STEP_PATIENCE", 2, #all shot

                "SOLVER.CHECKPOINT_PER_EPOCH", 1.0, 

                "SOLVER.AUTO_TERMINATE_PATIENCE", 8, #few shot
                # "SOLVER.AUTO_TERMINATE_PATIENCE", 4, #all shot

                "SOLVER.MODEL_EMA", 0.0,

                "SOLVER.TUNING_HIGHLEVEL_OVERRIDE", "linear_prob",

            ]
            
else:
    print("Error: have to select ft_type", file=sys.stderr)
    assert False

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
args.distributed = num_gpus > 1

if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )

cfg.local_rank = args.local_rank
cfg.num_gpus = num_gpus

cfg.merge_from_file(args.config_file)
print(cfg)
print("args.opts", args.opts)
cfg.merge_from_list(args.opts)


# output_dir = cfg.OUTPUT_DIR
output_dir = output_path+"/1st_OUTPUT/" + dir_name + "/" + ft_type + "/" + str(shot_num) + "_shot/" #tomiya

if os.path.exists(output_dir):
    print("Error: fine-tuning weight exists", file=sys.stderr)
    assert False

if output_dir:
    mkdir(output_dir)

logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
logger.info("Using {} GPUs".format(num_gpus))
logger.info(args)

#logger.info("Collecting env info (might take some time)")
#logger.info("\n" + collect_env_info())

logger.info("Loaded configuration file {}".format(args.config_file))
with open(args.config_file, "r") as cf:
    config_str = "\n" + cf.read()
    logger.info(config_str)
logger.info("Running with config:\n{}".format(cfg))

output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
logger.info("Saving config into: {}".format(output_config_path))
# save overloaded model config in the output directory
save_config(cfg, output_config_path)

ft_configs = []
if args.ft_tasks:
    for ft_file in args.ft_tasks.split(","):
        for file in sorted(glob.glob(ft_file)):
            ft_configs.append(file)
else:
    ft_configs = [args.config_file]

shuffle_seeds = []
if args.shuffle_seeds:
    shuffle_seeds = [int(seed) for seed in args.shuffle_seeds.split(',')]
else:
    shuffle_seeds = [None]

model = None
for task_id, ft_cfg in enumerate(ft_configs, 1):
    for shuffle_seed in shuffle_seeds:
        cfg_ = cfg.clone()
        cfg_.defrost()
        cfg_.merge_from_file(ft_cfg)
        cfg_.merge_from_list(args.opts)
        ft_output_dir = output_dir + '/ft_task_{}'.format(task_id)

        if args.custom_shot_and_epoch_and_general_copy:
            custom_shot = int(args.custom_shot_and_epoch_and_general_copy.split("_")[0])
            custom_epoch = int(args.custom_shot_and_epoch_and_general_copy.split("_")[1])
            custom_copy = int(args.custom_shot_and_epoch_and_general_copy.split("_")[2])
            cfg_.SOLVER.MAX_EPOCH = custom_epoch
            cfg_.DATASETS.GENERAL_COPY = custom_copy
            if args.use_prepared_data:
                if custom_shot != 0: # 0 means full data training
                    cfg_.DATASETS.TRAIN = ("{}_{}_{}".format(cfg_.DATASETS.TRAIN[0], custom_shot, cfg_.DATASETS.SHUFFLE_SEED), )
                    try:
                        custom_shot_val = int(args.custom_shot_and_epoch_and_general_copy.split("_")[3])
                    except:
                        custom_shot_val = custom_shot
                    cfg_.DATASETS.TEST = ("{}_{}_{}".format(cfg_.DATASETS.TEST[0], custom_shot_val, cfg_.DATASETS.SHUFFLE_SEED), )
                    if custom_shot_val == 1 or custom_shot_val == 3:
                        cfg_.DATASETS.GENERAL_COPY_TEST = 4 # to avoid less images than GPUs
            else:
                cfg_.DATASETS.FEW_SHOT = custom_shot
        else:
            custom_shot = None
            custom_epoch = None

        if shuffle_seed is not None:
            cfg_.DATASETS.SHUFFLE_SEED = shuffle_seed
            ft_output_dir = ft_output_dir + '_seed_{}'.format(shuffle_seed)

        # Remerge to make sure that the command line arguments are prioritized
        cfg_.merge_from_list(args.opts)
        if "last_checkpoint" in cfg_.MODEL.WEIGHT:
            with open(cfg_.MODEL.WEIGHT.replace("model_last_checkpoint.pth", "last_checkpoint"), "r") as f:
                last_checkpoint = f.read()
            cfg_.MODEL.WEIGHT = cfg_.MODEL.WEIGHT.replace("model_last_checkpoint.pth", last_checkpoint)
            print("cfg.MODEL.WEIGHT ", cfg_.MODEL.WEIGHT)

        mkdir(ft_output_dir)
        cfg_.OUTPUT_DIR = ft_output_dir

        tuning_highlevel_override(cfg_)
        cfg_.freeze()

        logger.info("Loaded fine-tune configuration file {}".format(ft_cfg))
        with open(ft_cfg, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)

        output_config_path = os.path.join(ft_output_dir, 'config.yml')
        print("Saving config into: {}".format(output_config_path))
        # save config here because the data loader will make some changes
        save_config(cfg_, output_config_path)
        logger.info("Training {}".format(ft_cfg))

(cfg, local_rank, distributed,
 zero_shot,
 skip_optimizer_resume,
 save_config_path)=(
 cfg_,  args.local_rank, args.distributed, 
 args.skip_train or custom_shot == 10000, 
 args.skip_optimizer_resume,
 output_config_path)

data_loader = make_data_loader(
    cfg,
    is_train=True,
    is_distributed=distributed,
    start_iter=0 #<TODO> Sample data from resume is disabled, due to the conflict with max_epoch
)

if cfg.TEST.DURING_TRAINING:
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    data_loaders_val = data_loaders_val[0]
else:
    data_loaders_val = None

model = build_detection_model(cfg)
device = torch.device(cfg.MODEL.DEVICE)
model.to(device)

if cfg.MODEL.LINEAR_PROB:
    assert cfg.MODEL.BACKBONE.FREEZE, "For linear probing, backbone should be frozen!"
    if hasattr(model.backbone, 'fpn'):
        assert cfg.MODEL.FPN.FREEZE, "For linear probing, FPN should be frozen!"
if cfg.MODEL.BACKBONE.FREEZE:
    for p in model.backbone.body.parameters():
        p.requires_grad = False
if cfg.MODEL.FPN.FREEZE:
    for p in model.backbone.fpn.parameters():
        p.requires_grad = False
if cfg.MODEL.RPN.FREEZE:
    for p in model.rpn.parameters():
        p.requires_grad = False
if cfg.MODEL.LINEAR_PROB:
    if model.rpn is not None:
        for key, p in model.rpn.named_parameters():
            if not ('bbox_pred' in key or 'cls_logits' in key or 'centerness' in key or 'cosine_scale' in key or 'dot_product_projection_text' in key or 'head.log_scale' in key or 'head.bias_lang' in key or 'head.bias0' in key):
                p.requires_grad = False
    if model.roi_heads is not None:
        for key, p in model.roi_heads.named_parameters():
            if not ('bbox_pred' in key or 'cls_logits' in key or 'centerness' in key or 'cosine_scale' in key or 'dot_product_projection_text' in key or 'head.log_scale' in key or 'head.bias_lang' in key or 'head.bias0' in key):
                p.requires_grad = False
if cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER:
    if model.rpn is not None:
        for key, p in model.rpn.named_parameters():
            if 'tunable_linear' in key:
                p.requires_grad = True

optimizer = make_optimizer(cfg, model)
scheduler = make_lr_scheduler(cfg, optimizer)

if distributed:
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank,
        # this should be removed if we update BatchNorm stats
        broadcast_buffers=False,
        find_unused_parameters=cfg.SOLVER.FIND_UNUSED_PARAMETERS
    )

arguments = {}
arguments["iteration"] = 0

output_dir = cfg.OUTPUT_DIR

save_to_disk = get_rank() == 0

checkpointer = DetectronCheckpointer(
    cfg, model, optimizer, scheduler, output_dir, save_to_disk
)
if checkpointer.has_checkpoint():
    extra_checkpoint_data = checkpointer.load(skip_optimizer=skip_optimizer_resume)
    arguments.update(extra_checkpoint_data)
else:
    state_dict = checkpointer._load_file(try_to_find(cfg.MODEL.WEIGHT))
    checkpointer._load_model(state_dict)

checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
meters = MetricLogger(delimiter="  ")

if zero_shot:
    assert False

if is_main_process():
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name, " : Not Frozen")
        else:
            print(name, " : Frozen")
report_freeze_options(cfg)

(cfg,model, data_loader, optimizer, scheduler,
 checkpointer, device, checkpoint_period,
 arguments, val_data_loader, meters)=(
 cfg, model, data_loader, optimizer, scheduler,
 checkpointer, device, checkpoint_period,
 arguments, data_loaders_val, meters)

logger = logging.getLogger("maskrcnn_benchmark.trainer")
logger.info("Start training")
# meters = MetricLogger(delimiter="  ")
max_iter = len(data_loader)
start_iter = arguments["iteration"]
model.train()
model_ema = None
if cfg.SOLVER.MODEL_EMA > 0:
    model_ema = ModelEma(model, decay=cfg.SOLVER.MODEL_EMA)
start_training_time = time.time()
end = time.time()

if cfg.SOLVER.USE_AMP:
    scaler = GradScaler()

global_rank = get_rank()

if cfg.SOLVER.CHECKPOINT_PER_EPOCH != -1 and cfg.SOLVER.MAX_EPOCH >= 1:
    checkpoint_period = len(data_loader) * cfg.SOLVER.CHECKPOINT_PER_EPOCH // cfg.SOLVER.MAX_EPOCH

if global_rank <= 0 and cfg.SOLVER.MAX_EPOCH >= 1:
    print("Iter per epoch ", len(data_loader) // cfg.SOLVER.MAX_EPOCH )

if cfg.SOLVER.AUTO_TERMINATE_PATIENCE != -1:
    patience_counter = 0
    previous_best = 0.0

# Adapt the weight decay
if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):
    milestone_target = 0
    for i, milstone in enumerate(list(scheduler.milestones)):
        if scheduler.last_epoch >= milstone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
            milestone_target = i+1

for iteration, (images, targets, idxs, positive_map, positive_map_eval, greenlight_map) in enumerate(data_loader, start_iter):
    
    nnegative = sum(len(target) < 1 for target in targets)
    nsample = len(targets)

    data_time = time.time() - end
    iteration = iteration + 1
    arguments["iteration"] = iteration

    images = images.to(device)
    captions = None
    try:
        targets = [target.to(device) for target in targets]
        captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
    except:
        pass

    if cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
        if hasattr(model, "module"):
            model.module.language_backbone.eval()
        else:
            model.language_backbone.eval()

    if cfg.SOLVER.USE_AMP:
        with autocast():
            if len(captions) > 0:
                loss_dict = model(images, targets, captions, positive_map, greenlight_map = greenlight_map)
            else:
                loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        if torch.isnan(losses) or torch.isinf(losses):
            logging.error("NaN encountered, ignoring")
            losses[losses != losses] = 0
        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        # logging.error(f"Weights updated. Loss is {losses.item()}")
    else:
        if len(captions) > 0:
            loss_dict = model(images, targets, captions, positive_map)
        else:
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        if torch.isnan(losses) or torch.isinf(losses):
            losses[losses != losses] = 0
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        scheduler.step()

    # Adapt the weight decay: only support multiStepLR
    if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):
        if milestone_target < len(scheduler.milestones):
            next_milestone = list(scheduler.milestones)[milestone_target]
        else:
            next_milestone = float('inf')
        if scheduler.last_epoch >= next_milestone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
            gamma = scheduler.gamma
            logger.info("Drop the weight decay by {}!".format(gamma))
            for param in optimizer.param_groups:
                if 'weight_decay' in param:
                    param['weight_decay'] *= gamma
            # move the target forward
            milestone_target += 1

    # reduce losses over all GPUs for logging purposes
    loss_dict_reduced = reduce_loss_dict(loss_dict)
    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
    meters.update(loss=losses_reduced, **loss_dict_reduced)
    if model_ema is not None:
        model_ema.update(model)
        arguments["model_ema"] = model_ema.state_dict()

    batch_time = time.time() - end
    end = time.time()
    meters.update(time=batch_time, data=data_time)
    eta_seconds = meters.time.global_avg * (max_iter - iteration)
    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
    
    if iteration % 20 == 0 or iteration == max_iter:
    # if iteration % 1 == 0 or iteration == max_iter:
        #logger.info(
        if global_rank <= 0:
            print(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "wd: {wd:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    wd=optimizer.param_groups[0]["weight_decay"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
    if val_data_loader and (iteration % checkpoint_period == 0 or iteration == max_iter):
        if is_main_process():
            print("Evaluating")
        eval_result = 0.0
        model.eval()
        if cfg.SOLVER.TEST_WITH_INFERENCE:
            with torch.no_grad():
                try:
                    _model = model.module
                except:
                    _model = model
                _result = inference(
                    model = _model,
                    data_loader = val_data_loader,
                    dataset_name="val",
                    device=device,
                    expected_results=cfg.TEST.EXPECTED_RESULTS,
                    expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                    output_folder=None,
                    cfg=cfg,
                    verbose=False
                )
                if is_main_process():
                    eval_result = _result[0].results['bbox']['AP']
        else:
            results_dict = {}
            cpu_device = torch.device("cpu")
            for i, batch in enumerate(val_data_loader):
                images, targets, image_ids, positive_map, *_ = batch
                with torch.no_grad():
                    images = images.to(device)
                    if positive_map is None:
                        output = model(images)
                    else:
                        captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
                        output = model(images, captions, positive_map)
                    output = [o.to(cpu_device) for o in output]
                results_dict.update(
                    {img_id: result for img_id, result in zip(image_ids, output)}
                )
            all_predictions = all_gather(results_dict)
            if is_main_process():
                predictions = {}
                for p in all_predictions:
                    predictions.update(p)
                predictions = [predictions[i] for i in list(sorted(predictions.keys()))]
                eval_result, _ = evaluate(val_data_loader.dataset, predictions, output_folder=None,
                                        box_only=cfg.DATASETS.CLASS_AGNOSTIC)
                if cfg.DATASETS.CLASS_AGNOSTIC:
                    eval_result = eval_result.results['box_proposal']['AR@100']
                else:
                    eval_result = eval_result.results['bbox']['AP']
        model.train()

        if model_ema is not None and cfg.SOLVER.USE_EMA_FOR_MONITOR:
            model_ema.ema.eval()
            results_dict = {}
            cpu_device = torch.device("cpu")
            for i, batch in enumerate(val_data_loader):
                images, targets, image_ids, positive_map, positive_map_eval = batch
                with torch.no_grad():
                    images = images.to(device)
                    if positive_map is None:
                        output = model_ema.ema(images)
                    else:
                        captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
                        output = model_ema.ema(images, captions, positive_map)
                    output = [o.to(cpu_device) for o in output]
                results_dict.update(
                    {img_id: result for img_id, result in zip(image_ids, output)}
                )
            all_predictions = all_gather(results_dict)
            if is_main_process():
                predictions = {}
                for p in all_predictions:
                    predictions.update(p)
                predictions = [predictions[i] for i in list(sorted(predictions.keys()))]
                eval_result, _ = evaluate(val_data_loader.dataset, predictions, output_folder=None,
                                          box_only=cfg.DATASETS.CLASS_AGNOSTIC)
                if cfg.DATASETS.CLASS_AGNOSTIC:
                    eval_result = eval_result.results['box_proposal']['AR@100']
                else:
                    eval_result = eval_result.results['bbox']['AP']

        arguments.update(eval_result=eval_result)

        if cfg.SOLVER.USE_AUTOSTEP:
            eval_result = all_gather(eval_result)[0] #broadcast_data([eval_result])[0]
            # print("Rank {} eval result gathered".format(cfg.local_rank), eval_result)
            scheduler.step(eval_result)

        if cfg.SOLVER.AUTO_TERMINATE_PATIENCE != -1:
            if eval_result < previous_best:
                patience_counter += 1
            else:
                patience_counter = 0
                previous_best = eval_result
                checkpointer.save("model_best", **arguments)
            print("Previous Best", previous_best, "Patience Counter", patience_counter, "Eval Result", eval_result)
            if patience_counter >= cfg.SOLVER.AUTO_TERMINATE_PATIENCE:
                if is_main_process():
                    print("\n\n\n\nAuto Termination at {}, current best {}\n\n\n".format(iteration, previous_best))
                break

    if iteration % checkpoint_period == 0:
        checkpointer.save("model_{:07d}".format(iteration), **arguments)
    if iteration == max_iter:
        checkpointer.save("model_final", **arguments)
        break

total_training_time = time.time() - start_training_time
total_time_str = str(datetime.timedelta(seconds=total_training_time))
logger.info(
    "Total training time: {} ({:.4f} s / it)".format(
        total_time_str, total_training_time / (max_iter)
    )
)
