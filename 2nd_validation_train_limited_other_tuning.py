import os
os.chdir('./GLIP')

MAX_TRAIN_ANNOTATIONS_PER_CATEGORY = 400
MAX_VAL_ANNOTATIONS_PER_CATEGORY = MAX_TRAIN_ANNOTATIONS_PER_CATEGORY // 3  # trainの1/3

dir_name = f"20250411共有_5cat_train_{MAX_TRAIN_ANNOTATIONS_PER_CATEGORY}_val_{MAX_VAL_ANNOTATIONS_PER_CATEGORY}_valid"
out_dir_name = f"20250411共有_5cat_train_{MAX_TRAIN_ANNOTATIONS_PER_CATEGORY}_val_{MAX_VAL_ANNOTATIONS_PER_CATEGORY}_valid"
task_config = f"../../bridge_data/configs/20250411共有_5cat_train_{MAX_TRAIN_ANNOTATIONS_PER_CATEGORY}_val_{MAX_VAL_ANNOTATIONS_PER_CATEGORY}_test.yaml"
test_name = "test"

# ft_type = "full"
# ft_type = "prompt"
ft_type = "linear_prob"
shot_num = "all"
test_seed = "1"

output_path = "../"

#　出力画像を保存するかどうかのフラグ 1であれば保存 0であれば保存しない
output_image_flg=1

# 検出の出力を表示する際のスコアの閾値
out_thresh = 0.5

output_image_dir = f"../image_output/20250411共有_5cat_train_{MAX_TRAIN_ANNOTATIONS_PER_CATEGORY}_val_{MAX_VAL_ANNOTATIONS_PER_CATEGORY}_test"

if shot_num=="zero":
    config_file = "./configs/pretrain/glip_Swin_L.yaml"
    weight = "./MODEL/glip_large_model.pth"
else:
    config_file = output_path + "/1st_OUTPUT/" + dir_name + "/" + ft_type + "/" + str(shot_num) + "_shot/ft_task_" + test_seed + "/config.yml"
    weight = output_path + "/1st_OUTPUT/" + dir_name + "/" + ft_type + "/" + str(shot_num) + "_shot/ft_task_" + test_seed + "/model_best.pth"

log_dir = output_path + "/2nd_RESULT/" + out_dir_name + "/result_" + test_name + "/" + ft_type + "/" + str(shot_num) + "_shot/"


# 必要なパッケージのimport（前準備）
import argparse
import os
import torch
import functools
import io
import datetime
import torch.distributed as dist
import logging
import time
import re
import pdb
import copy
import csv
import matplotlib.pyplot as plt
import json
import numpy as np
import yaml

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.stats import get_model_complexity_info
from tqdm import tqdm
from collections import defaultdict
from maskrcnn_benchmark.data.datasets.evaluation import evaluate, im_detect_bbox_aug
from maskrcnn_benchmark.utils.comm import is_main_process
from maskrcnn_benchmark.utils.comm import all_gather
from maskrcnn_benchmark.utils.comm import synchronize
from maskrcnn_benchmark.data.datasets.evaluation.flickr.flickr_eval import FlickrEvaluator
from collections import defaultdict
from pycocotools import mask as maskUtils
from maskrcnn_benchmark.engine.inference import *
from maskrcnn_benchmark.data import datasets
from maskrcnn_benchmark.data.datasets.evaluation.coco import coco_evaluation
from maskrcnn_benchmark.data.datasets.evaluation.voc import voc_evaluation
from maskrcnn_benchmark.data.datasets.evaluation.vg import vg_evaluation
from maskrcnn_benchmark.data.datasets.evaluation.box_aug import im_detect_bbox_aug
from maskrcnn_benchmark.data.datasets.evaluation.od_to_grounding import od_to_grounding_evaluation
from maskrcnn_benchmark.data.datasets.evaluation.coco.coco_eval import do_coco_evaluation
from IPython.display import clear_output
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
import cv2

class args:
    
    config_file = config_file
    weight = weight
    task_config = task_config
    
    local_rank = 0
    world_size = 1
    dist_url = "env://"
    
    opts = [
        "TEST.IMS_PER_BATCH", 1,
        "SOLVER.IMS_PER_BATCH", 1,
        "TEST.EVAL_TASK", "detection",
        "DATASETS.TRAIN_DATASETNAME_SUFFIX", "_grounding",
        "DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE", False,
        "DATASETS.USE_OVERRIDE_CATEGORY", True,
        "DATASETS.USE_CAPTION_PROMPT", False #tomiya
    ]

class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all']
        self.useCats = 1

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None

# Helper

def init_distributed_mode(args):
    """Initialize distributed training, if appropriate"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    #args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print("| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True)

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank,
        timeout=datetime.timedelta(0, 7200)
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions

def prepare_for_coco_detection(predictions, dataset):
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]["width"]
        image_height = dataset.coco.imgs[original_id]["height"]
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert("xywh")

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        for k, box in enumerate(boxes):
            if labels[k] in dataset.contiguous_category_id_to_json_id:
                coco_results.append(
                    {
                        "image_id": original_id,
                        "category_id": dataset.contiguous_category_id_to_json_id[labels[k]],
                        "bbox": box,
                        "score": scores[k],
                    })

    return coco_results


def prepare_for_coco_segmentation(predictions, dataset):
    import pycocotools.mask as mask_util
    import numpy as np

    masker = Masker(threshold=0.5, padding=1)
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in tqdm(enumerate(predictions)):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]["width"]
        image_height = dataset.coco.imgs[original_id]["height"]
        prediction = prediction.resize((image_width, image_height))
        masks = prediction.get_field("mask")
        # t = time.time()
        # Masker is necessary only if masks haven't been already resized.
        if list(masks.shape[-2:]) != [image_height, image_width]:
            masks = masker(masks.expand(1, -1, -1, -1, -1), prediction)
            masks = masks[0]
        # logger.info('Time mask: {}'.format(time.time() - t))
        # prediction = prediction.convert('xywh')

        # boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        # rles = prediction.get_field('mask')

        rles = [
            mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "segmentation": rle,
                    "score": scores[k],
                }
                for k, rle in enumerate(rles)
            ]
        )
    return coco_results


def prepare_for_coco_keypoint(predictions, dataset):
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction.bbox) == 0:
            continue

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]['width']
        image_height = dataset.coco.imgs[original_id]['height']
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert('xywh')

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field('scores').tolist()
        labels = prediction.get_field('labels').tolist()
        keypoints = prediction.get_field('keypoints')
        keypoints = keypoints.resize((image_width, image_height))
        keypoints = keypoints.to_coco_format()

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend([{
            'image_id': original_id,
            'category_id': mapped_labels[k],
            'keypoints': keypoint,
            'score': scores[k]} for k, keypoint in enumerate(keypoints)])
    return coco_results


# inspired from Detectron
def evaluate_box_proposals(
        predictions, dataset, thresholds=None, area="all", limit=None
):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]["width"]
        image_height = dataset.coco.imgs[original_id]["height"]
        prediction = prediction.resize((image_width, image_height))

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        if prediction.has_field("objectness"):
            inds = prediction.get_field("objectness").sort(descending=True)[1]
        else:
            inds = prediction.get_field("scores").sort(descending=True)[1]
        prediction = prediction[inds]

        ann_ids = dataset.coco.getAnnIds(imgIds=original_id)
        anno = dataset.coco.loadAnns(ann_ids)
        gt_boxes = [obj["bbox"] for obj in anno if obj["iscrowd"] == 0]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = BoxList(gt_boxes, (image_width, image_height), mode="xywh").convert(
            "xyxy"
        )
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if len(prediction) == 0:
            continue

        if limit is not None and len(prediction) > limit:
            prediction = prediction[:limit]

        overlaps = boxlist_iou(prediction, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(prediction), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)

    if len(gt_overlaps) == 0:
        return {
            "ar": torch.zeros(1),
            "recalls": torch.zeros(1),
            "thresholds": thresholds,
            "gt_overlaps": gt_overlaps,
            "num_pos": num_pos,
        }

    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


def evaluate_predictions_on_coco(
        coco_gt, coco_results, json_result_file, iou_type="bbox"
):
    import json

    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO()

    # coco_dt = coco_gt.loadRes(coco_results)
    if iou_type == 'keypoints':
        coco_gt = filter_valid_keypoints(coco_gt, coco_dt)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    if iou_type == 'bbox':
        summarize_per_category(coco_eval, json_result_file.replace('.json', '.csv'))
    return coco_eval


def summarize_per_category(coco_eval, csv_output=None):
    '''
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    '''

    def _summarize(iouThr=None, areaRng='all', maxDets=100):
        p = coco_eval.params
        titleStr = 'Average Precision'
        typeStr = '(AP)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)
        result_str = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ], '. \
            format(titleStr, typeStr, iouStr, areaRng, maxDets)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        # dimension of precision: [TxRxKxAxM]
        s = coco_eval.eval['precision']
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
            # cacluate AP(average precision) for each category
            num_classes = len(p.catIds)
            avg_ap = 0.0
            for i in range(0, num_classes):
                result_str += '{}, '.format(np.mean(s[:, :, i, :]))
                avg_ap += np.mean(s[:, :, i, :])
            result_str += ('{} \n'.format(avg_ap / num_classes))
        return result_str

    id2name = {}
    for _, cat in coco_eval.cocoGt.cats.items():
        id2name[cat['id']] = cat['name']
    title_str = 'metric, '
    for cid in coco_eval.params.catIds:
        title_str += '{}, '.format(id2name[cid])
    title_str += 'avg \n'

    results = [title_str]
    results.append(_summarize())
    results.append(_summarize(iouThr=.5, maxDets=coco_eval.params.maxDets[2]))
    results.append(_summarize(areaRng='small', maxDets=coco_eval.params.maxDets[2]))
    results.append(_summarize(areaRng='medium', maxDets=coco_eval.params.maxDets[2]))
    results.append(_summarize(areaRng='large', maxDets=coco_eval.params.maxDets[2]))

    with open(csv_output, 'w') as f:
        for result in results:
            f.writelines(result)


def filter_valid_keypoints(coco_gt, coco_dt):
    kps = coco_dt.anns[1]['keypoints']
    for id, ann in coco_gt.anns.items():
        ann['keypoints'][2::3] = [a * b for a, b in zip(ann['keypoints'][2::3], kps[2::3])]
        ann['num_keypoints'] = sum(ann['keypoints'][2::3])
    return coco_gt

def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre

def calculate_iou(bbox1, bbox2):
    # bbox1とbbox2の座標情報を取得
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # 交差部分の座標を計算
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    x_intersection_end = min(x1 + w1, x2 + w2)
    y_intersection_end = min(y1 + h1, y2 + h2)

    # 交差部分の幅と高さを計算
    intersection_width = max(0, x_intersection_end - x_intersection)
    intersection_height = max(0, y_intersection_end - y_intersection)

    # 交差部分の面積を計算
    intersection_area = intersection_width * intersection_height

    # それぞれの境界ボックスの面積を計算
    area_bbox1 = w1 * h1
    area_bbox2 = w2 * h2

    # IoUを計算
    iou = intersection_area / (area_bbox1 + area_bbox2 - intersection_area)

    return iou

# Step1

num_gpus = 1
distributed = False

cfg.local_rank = args.local_rank
cfg.num_gpus = num_gpus

cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

if log_dir:
    mkdir(log_dir)

logger = setup_logger("maskrcnn_benchmark", log_dir, get_rank())
logger.info(args)
logger.info("Using {} GPUs".format(num_gpus))
logger.info(cfg)

# logger.info("Collecting env info (might take some time)")
# logger.info("\n" + collect_env_info())

model = build_detection_model(cfg)
model.to(cfg.MODEL.DEVICE)

model.rpn.box_selector_test.pre_nms_thresh=.0001 # custom

checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
if args.weight:
    _ = checkpointer.load(args.weight, force=True)
else:
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

all_task_configs = args.task_config.split(",")

task_config = all_task_configs[0]

cfg_ = cfg.clone()
cfg_.defrost()
cfg_.merge_from_file(task_config)
cfg_.merge_from_list(args.opts)
iou_types = ("bbox",)
if cfg_.MODEL.MASK_ON:
    iou_types = iou_types + ("segm",)
if cfg_.MODEL.KEYPOINT_ON:
    iou_types = iou_types + ("keypoints",)
dataset_names = cfg_.DATASETS.TEST
if isinstance(dataset_names[0], (list, tuple)):
    dataset_names = [dataset for group in dataset_names for dataset in group]
output_folders = [None] * len(dataset_names)

for idx, dataset_name in enumerate(dataset_names):
    # output_folder = os.path.join(log_dir, "inference", dataset_name) #tomiya todo here
    output_folder = log_dir
    mkdir(output_folder)
    output_folders[idx] = output_folder
data_loaders_val = make_data_loader(cfg_, is_train=False, is_distributed=distributed)

# Inference

output_folder = output_folders[0]
dataset_name = dataset_names[0]
data_loader_val = data_loaders_val[0]

(model,data_loader, dataset_name,
iou_types, box_only, device,
expected_results, expected_results_sigma_tol,
output_folder, cfg, verbose)=\
(model,data_loader_val, dataset_name,
 iou_types, cfg_.MODEL.RPN_ONLY and (cfg_.MODEL.RPN_ARCHITECTURE == "RPN" or cfg_.DATASETS.CLASS_AGNOSTIC), cfg_.MODEL.DEVICE,
 cfg_.TEST.EXPECTED_RESULTS, cfg_.TEST.EXPECTED_RESULTS_SIGMA_TOL,
 output_folder, cfg_, True)

# convert to a torch.device for efficiency
try:
    device = torch.device(device)
except:
    device = device
num_devices = (
    torch.distributed.get_world_size()
    if torch.distributed.is_initialized()
    else 1
)

logger = logging.getLogger("maskrcnn_benchmark.inference")
dataset = data_loader.dataset
if verbose:
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
start_time = time.time()

task = cfg.TEST.EVAL_TASK

if task == "detection":
    all_queries, all_positive_map_label_to_token = create_queries_and_maps_from_dataset(dataset, cfg)
elif task == "grounding":
    all_queries = [None]
    all_positive_map_label_to_token = [None]
else:
    assert(0)

categories = dataset.categories()
#one_hot = dataset.one_hot

labels = []
label_list = []
keys = list(categories.keys())
keys.sort()
for i in keys:
    labels.append(i)
    label_list.append(categories[i])

if cfg.TEST.CHUNKED_EVALUATION != -1:
    labels = chunks(labels, cfg.TEST.CHUNKED_EVALUATION)
    label_list = chunks(label_list, cfg.TEST.CHUNKED_EVALUATION)
else:
    labels = [labels]
    label_list = [label_list]

all_queries = []
all_positive_map_label_to_token = []

for i in range(len(labels)):
    labels_i = labels[i]
    label_list_i = label_list[i]
    query_i, positive_map_label_to_token_i = create_queries_and_maps(
        labels_i, label_list_i, additional_labels = cfg.DATASETS.SUPRESS_QUERY if cfg.DATASETS.USE_SUPRESS_QUERY else None, cfg = cfg)

    all_queries.append(query_i)
    all_positive_map_label_to_token.append(positive_map_label_to_token_i)
print("All queries", all_queries)


'''
Build Dataset Sepecific Evaluator
'''
if "flickr" in cfg.DATASETS.TEST[0]:
    evaluator = build_flickr_evaluator(cfg)
elif "lvis" in cfg.DATASETS.TEST[0]:
    evaluator = build_lvis_evaluator(dataset.ann_file, fixed_ap=not cfg.DATASETS.LVIS_USE_NORMAL_AP)
else:
    evaluator = None

model.eval()
results_dict = {}
cpu_device = torch.device("cpu")
if verbose:
    _iterator = tqdm(data_loader)
else:
    _iterator = data_loader

for i, batch in enumerate(_iterator):
    if i == cfg.TEST.SUBSET:
        break
    images, targets, image_ids, *_ = batch

    all_output = []
    mdetr_style_output = []
    break

images = images.to(device)
query_time = len(all_queries)

query_i = 0

if not isinstance(targets[0], dict): # For LVIS dataset and datasets directly copied from MDETR
    targets = [target.to(device) for target in targets]
'''
different datasets seem to have different data format... For LVIS dataset, the target is a dictionary, while for modulatedDataset such as COCO/Flickr, the target is a BoxList
'''

if task == "detection":
    captions = [all_queries[query_i] for ii in range(len(targets))]
    positive_map_label_to_token = all_positive_map_label_to_token[query_i]

output = model(images, captions=captions, positive_map=positive_map_label_to_token)
output = [o.to(cpu_device) for o in output]

all_output.append(output)

for i, batch in enumerate(_iterator):
    if i == cfg.TEST.SUBSET:
        break
    images, targets, image_ids, *_ = batch

    all_output = []
    mdetr_style_output = []
    with torch.no_grad():
        images = images.to(device)
        query_time = len(all_queries)

        for query_i in range(query_time):
            if not isinstance(targets[0], dict): # For LVIS dataset and datasets directly copied from MDETR
                targets = [target.to(device) for target in targets]
            '''
            different datasets seem to have different data format... For LVIS dataset, the target is a dictionary, while for modulatedDataset such as COCO/Flickr, the target is a BoxList
            '''

            if task == "detection":
                captions = [all_queries[query_i] for ii in range(len(targets))]
                positive_map_label_to_token = all_positive_map_label_to_token[query_i]

            output = model(images, captions=captions, positive_map=positive_map_label_to_token)
            output = [o.to(cpu_device) for o in output]

            all_output.append(output)
    if evaluator is not None:
        evaluator.update(mdetr_style_output)
    else:
        output = [[row[_i] for row in all_output] for _i in range(len(all_output[0]))]
        for index, i in enumerate(output):
            output[index] = i[0].concate_box_list(i)

        results_dict.update({img_id: result for img_id, result in zip(image_ids, output)})
    
    # GPUメモリを解放
    torch.cuda.empty_cache()

if evaluator is not None:
    evaluator.synchronize_between_processes()
    try:
        evaluator.accumulate()
    except:
        print("Evaluator has no accumulation, skipped...")
    score = evaluator.summarize()
    print(score)
    import maskrcnn_benchmark.utils.mdetr_dist as dist
    if is_main_process():
        if "flickr" in cfg.DATASETS.TEST[0]:
            write_flickr_results(score, output_file_name=os.path.join(output_folder, "bbox.csv"))
        elif "lvis" in cfg.DATASETS.TEST[0]:
            write_lvis_results(score, output_file_name=os.path.join(output_folder, "bbox.csv"))
    try:
        torch.distributed.barrier()
    except:
        print("Default process group is not initialized")
    assert False

if evaluator is not None:
    predictions = mdetr_style_output
else:
    predictions = results_dict

# wait for all processes to complete before measuring the time
synchronize()
total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=total_time))
logger.info(
    "Total inference time: {} ({} s / img per device, on {} devices)".format(
        total_time_str, total_time * num_devices / len(dataset), num_devices
    )
)

predictions = _accumulate_predictions_from_multiple_gpus(predictions)

print("Accumulated results")
if not is_main_process():
    assert False

if output_folder:
    torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

extra_args = dict(
    box_only=box_only,
    iou_types=iou_types,
    expected_results=expected_results,
    expected_results_sigma_tol=expected_results_sigma_tol,
)

(dataset, predictions, output_folder,\
 box_only, iou_types, expected_results,\
 expected_results_sigma_tol)=(dataset, predictions, output_folder,\
                              box_only, iou_types, expected_results,\
                              expected_results_sigma_tol)

class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        # TODO make it pretty
        return repr(self.results)


def check_expected_results(results, expected_results, sigma_tol):
    if not expected_results:
        return

    logger = logging.getLogger("maskrcnn_benchmark.inference")
    for task, metric, (mean, std) in expected_results:
        actual_val = results.results[task][metric]
        lo = mean - sigma_tol * std
        hi = mean + sigma_tol * std
        ok = (lo < actual_val) and (actual_val < hi)
        msg = (
            "{} > {} sanity check (actual vs. expected): "
            "{:.3f} vs. mean={:.4f}, std={:.4}, range=({:.4f}, {:.4f})"
        ).format(task, metric, actual_val, mean, std, lo, hi)
        if not ok:
            msg = "FAIL: " + msg
            logger.error(msg)
        else:
            msg = "PASS: " + msg
            logger.info(msg)

logger = logging.getLogger("maskrcnn_benchmark.inference")

logger.info("Preparing results for COCO format")
coco_results = {}
if "bbox" in iou_types:
    logger.info("Preparing bbox results")
    if dataset.coco is None:
        coco_results["bbox"] = prepare_for_tsv_detection(predictions, dataset)
    else:
        coco_results["bbox"] = prepare_for_coco_detection(predictions, dataset)
        
# Evaluation

(dataset, predictions, output_folder, kwargs)=(dataset, predictions, output_folder, extra_args)

args_eval = dict(
    dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs
)

(coco_gt, coco_results, json_result_file, iou_type) = (dataset.coco, coco_results["bbox"], "./test.csv", "bbox")

with open(json_result_file, "w") as f:
    json.dump(coco_results, f)

coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO()

cocoGt=None

num_images = len(dataset)

num_categories = len(positive_map_label_to_token)


#  Class COCOeval
class COCOeval:

    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                 for catId in catIds
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
             ]

        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def computeIoU_old(self, imgId, catId):
        print(str(imgId)+":"+str(catId))
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d,g,iscrowd)
        print(ious)
        return ious

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]; yd = d[1::3]
                if k1>0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                    dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
                e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
                if k1 > 0:
                    e=e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
                    
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]
        
        dtbbox = np.array([d['bbox'] for d in dt])
        #print(dtbbox)
        

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        
        
        imgId_list = [imgId]* len(dt)
        
        
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
                'dtBbox':       dtbbox, #here
                'image_id_list':imgId_list,
            }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        #R           = len(p.recThrs)
        
        R =  100*num_images #here
        
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T, R, K, A, M)) # -1 for the precision of absent categories
        recall      = -np.ones((T, K, A, M))
        scores      = -np.ones((T, R, K, A, M))
        recall_all  = -np.ones((T, R, K, A, M))#here
        bbox_all    = -np.ones((T, R, 4, K, A, M))#here
        imgId_all    = -np.ones((T, R, K, A, M))#here
        # print(bbox_all.shape)
        # (10, 63000, 4, 4, 1, 3)
        

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])
                    
                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    
                    
                    #dtb  = np.concatenate([e['dtBbox'][0:maxDet] for e in E])[inds]#here
                    
                    dtb_before = [e['dtBbox'][0:maxDet] for e in E]
                    
                    # dtb_after = [arr if arr.size > 0 else np.array([[-5,-5,-5,-5]]) for arr in dtb_before]
                    dtb_after = [arr for arr in dtb_before if arr.size > 0]
                    
                    dtb  = np.concatenate(dtb_after)[inds]#here
                    
                    dtId = np.concatenate([e['image_id_list'][0:maxDet] for e in E])[inds]
                    dtId = dtId.tolist()
                    
                                        
                    npig = np.count_nonzero(gtIg==0 )
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                    
                    
                    
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))
                        recall_np  = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        #tomiya
                        padded_pr = np.pad(np.array(pr), (0, R - len(np.array(pr))),  mode='constant', constant_values=-1)
                        padded_dtscore = np.pad(np.array(dtScoresSorted), (0, R - len(np.array(dtScoresSorted))),  mode='constant', constant_values=-1)
                        padded_rc = np.pad(np.array(rc), (0, R - len(np.array(rc))),  mode='constant', constant_values=-1)
                        padded_id = np.pad(np.array(dtId), (0, R - len(np.array(dtId))),  mode='constant', constant_values=-1)

                        padded_bbox0 = np.pad(np.array(dtb[:,0]), (0, R - len(np.array(dtb[:,0]))),  mode='constant', constant_values=-1)
                        padded_bbox1 = np.pad(np.array(dtb[:,1]), (0, R - len(np.array(dtb[:,1]))),  mode='constant', constant_values=-1)
                        padded_bbox2 = np.pad(np.array(dtb[:,2]), (0, R - len(np.array(dtb[:,2]))),  mode='constant', constant_values=-1)
                        padded_bbox3 = np.pad(np.array(dtb[:,3]), (0, R - len(np.array(dtb[:,3]))),  mode='constant', constant_values=-1)
                        bbox_all[t,:,0,k,a,m] = padded_bbox0
                        bbox_all[t,:,1,k,a,m] = padded_bbox1
                        bbox_all[t,:,2,k,a,m] = padded_bbox2
                        bbox_all[t,:,3,k,a,m] = padded_bbox3

                        precision[t,:,k,a,m] = padded_pr
                        scores[t,:,k,a,m] = padded_dtscore 
                        recall_all[t,:,k,a,m] = padded_rc
                        
                        imgId_all[t,:,k,a,m] = padded_id
                        
                    # print("t:"+str(t)+"k:"+str(k)+"a:"+str(a)+"m:"+str(m))
                    # print(scores.tolist()[0][0:2])
        
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
            'recall_all': recall_all,
            'bbox_all': bbox_all,
            'id_all': imgId_all,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']

                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
                
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
                
            scores_list = self.eval['scores']
            
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                scores_list = scores_list[t]
            scores_list = scores_list[:,:,:,aind,mind]
            
            recall_list = self.eval['recall_all']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                recall_list = recall_list[t]
            recall_list = recall_list[:,:,:,aind,mind]

            bbox_list = self.eval['bbox_all']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                bbox_list = bbox_list[t]
            bbox_list = bbox_list[:,:,:,:,aind,mind]
            
            id_list = self.eval['id_all']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                id_list = id_list[t]
            id_list = id_list[:,:,:,aind,mind]

            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            
            s_cat_all=[]
            recall_cat_all=[]
            scores_cat_all=[]
            bboxes_cat_all=[]
            id_cat_all=[]
            
            for cat_num in range(num_categories):
                s_cat = s[:,:,cat_num]
                s_cat_all.append(s_cat)    
                
            for cat_num in range(num_categories):
                recall_cat = recall_list[:,:,cat_num]
                recall_cat_all.append(recall_cat)
                
            for cat_num in range(num_categories):
                scores_cat = scores_list[:,:,cat_num]
                scores_cat_all.append(scores_cat)    
                
            for cat_num in range(num_categories):
                bboxes_cat = bbox_list[:,:,:,cat_num]
                bboxes_cat_all.append(bboxes_cat)    

            for cat_num in range(num_categories):
                id_cat = id_list[:,:,cat_num]
                id_cat_all.append(id_cat)    
   
            write_list_p=[]
            for cat_num in range(num_categories):
                tmp = s_cat_all[cat_num]
                tmp2 = tmp[0,:,0]
                write_list_p.append(tmp2.tolist())
            
            write_list_r=[]
            for cat_num in range(num_categories):
                tmp = recall_cat_all[cat_num]
                tmp2 = tmp[0,:,0]
                write_list_r.append(tmp2.tolist())

            write_list_sc=[]
            for cat_num in range(num_categories):
                tmp = scores_cat_all[cat_num]
                tmp2 = tmp[0,:,0]
                write_list_sc.append(tmp2.tolist())
                
            write_list_bbox=[]
            for cat_num in range(num_categories):
                tmp = bboxes_cat_all[cat_num]
                tmp2 = tmp[0,:,:,0]
                write_list_bbox.append(tmp2.tolist())
                
            write_list_id=[]
            for cat_num in range(num_categories):
                tmp = id_cat_all[cat_num]
                tmp2 = tmp[0,:,0]
                write_list_id.append(tmp2.tolist())
            
            with open(output_folder+'precision.csv', 'w') as file:
                writer = csv.writer(file, lineterminator='\n')
                writer.writerows(write_list_p)
                
            with open(output_folder+'recall.csv', 'w') as file:
                writer = csv.writer(file, lineterminator='\n')
                writer.writerows(write_list_r)
    
            with open(output_folder+'scores.csv', 'w') as file:
                writer = csv.writer(file, lineterminator='\n')
                writer.writerows(write_list_sc)
                
            with open(output_folder+'bboxes.csv', 'w') as file:
                writer = csv.writer(file, lineterminator='\n')
                writer.writerows(write_list_bbox)
                
            with open(output_folder+'imgId.csv', 'w') as file:
                writer = csv.writer(file, lineterminator='\n')
                writer.writerows(write_list_id)
            
            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            #stats[0] = _summarize(1)
            stats[0] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            # stats[1] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            # stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            # stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            # stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            # stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            # stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            # stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            # stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            # stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            # stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d,g,iscrowd)
        return ious


# COCOeval evaluation
coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
if iou_type == 'bbox':
    summarize_per_category(coco_eval, json_result_file.replace('.json', '.csv'))

with open(task_config, 'r') as yml:
    config = yaml.safe_load(yml)

json_path = config['DATASETS']['REGISTER']['test']['ann_file']

with open(json_path, 'r') as json_file:
    data = json.load(json_file)

cat_len = len(data['categories'])

cat_name=[]
for each in data['categories']:
    cat_name.append(each['name'])

with open(output_folder+'precision.csv') as file:
    pre_all = list(csv.reader(file))

with open(output_folder+'recall.csv') as file:
    rec_all = list(csv.reader(file))

with open(output_folder+'scores.csv') as file:
    sc_all = list(csv.reader(file))

with open(output_folder+'imgId.csv') as file:
    id_all = list(csv.reader(file))

with open(output_folder+'bboxes.csv') as file:
    bboxes_all = list(csv.reader(file))
    
pre = [[float(element) for element in row] for row in pre_all]

rec = [[float(element) for element in row] for row in rec_all]

sc = [[float(element) for element in row] for row in sc_all]

idid = [[float(element) for element in row] for row in id_all]

bbox=[]
for row in bboxes_all:
    inner_list = []
    for item in row:
        num_list = [float(val) for val in item.strip('[]').split(', ')]
        inner_list.append(num_list)
    bbox.append(inner_list)

cat_iou_all=[]
for cat_id in tqdm(range(cat_len)):
    cat_iou=[]
    for k, name in enumerate(idid[cat_id]):
        if name==-1:
            break
        iou_list=[0]*cat_len
        for each_id in range(cat_len):
            iou_each=[-1]
            for l, each in enumerate(data['annotations']):
                if each['image_id']==name and each['category_id']==each_id+1:
                    iou1 = calculate_iou(bbox[cat_id][k],  each['bbox'])
                    iou_each.append(iou1)
            iou_max = max(iou_each)
            iou_list[each_id]=iou_max
        cat_iou.append(iou_list)
    cat_iou_all.append(cat_iou)

with open(output_folder+'iou_all.csv', 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerows(cat_iou_all)

with open(output_folder+'iou_all.csv') as file:
    ioues_all = list(csv.reader(file))

print("Average Precision [%]")

for cat_id in range(cat_len):
    tail=pre[cat_id].count(-1)
    ap,mrec,mpre=voc_ap(rec[cat_id][:-tail],pre[cat_id][:-tail])
    print("---" + cat_name[cat_id] + "---")
    print(ap*100)

# ボックスの色を定義　6種類用意
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    
# 見える化の関数の定義
def plot_results(pil_img, boxes, labels, scores, figsize_input):
    font_size = 12 #フォントサイズを設定可能
    plt.figure(figsize=figsize_input)
    np_image = np.array(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    assert len(boxes) == len(labels)
    for (xmin, ymin, xmax, ymax), l, s, c in zip(boxes, labels, scores, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{l}: {s:0.2f}'
        ax.text(xmin, ymin-(font_size+2), text, fontsize=font_size, bbox=dict(facecolor='white', alpha=0.8))
    plt.imshow(np_image)
    plt.axis('off')
    
check_image_list = [tmp["id"] for tmp in data["images"]]

if output_image_flg==1:
    for imid in check_image_list:
        sc_list = []
        bb_list = []
        ct_list = []

        for cat_id_gt in range(cat_len):
            count = sum(1 for item in sc[cat_id_gt] if item > out_thresh)
            for k, each in enumerate(idid[cat_id_gt][0:count]):
                if each==imid:
                    x, y, width, height = bbox[cat_id_gt][k]
                    sc_list.append(sc[cat_id_gt][k])
                    bb_list.append([x, y, x+width, y+height])
                    ct_list.append(data['categories'][cat_id_gt]['name']) 

        for each in data['images']:
            if each['id']==imid:
                file_name = each['file_name']
                break
                
        bb_list = bb_list[::-1]
        ct_list = ct_list[::-1]
        sc_list = sc_list[::-1]

        each_image = cv2.imread("../../bridge_data/20250411共有/" + file_name)
        each_image = cv2.cvtColor(each_image, cv2.COLOR_BGR2RGB)

        # bbox_eachはbb_listが空でない場合のみ作成
        if bb_list:
            # 最初のボックスから座標を取得
            first_box = bb_list[0]
            x, y, x_max, y_max = first_box
            width = x_max - x
            height = y_max - y
            bbox_each = [x, y, x+width, y+height]

        image_height = each_image.shape[0]
        image_width = each_image.shape[1]

        figsize_input=(image_width/100, image_height/100)

        plot_results(each_image, bb_list, ct_list, sc_list, figsize_input)
        plt.savefig(output_image_dir + "/" + file_name.split(".")[0] + "_output.jpg", bbox_inches='tight', pad_inches=0.1, dpi=100)

# PR曲線描画の追加
print("\nPR曲線を描画しています...")

# PR曲線用の出力ディレクトリを作成
pr_curve_output_dir = output_folder + "pr_curves/"
os.makedirs(pr_curve_output_dir, exist_ok=True)

# 各カテゴリのPR曲線を描画
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# 全カテゴリをまとめたPR曲線用
plt.figure(figsize=(10, 8))

for cat_id in range(cat_len):
    # -1の値を除去（有効なデータのみ使用）
    tail = pre[cat_id].count(-1)
    valid_recall = rec[cat_id][:-tail] if tail > 0 else rec[cat_id]
    valid_precision = pre[cat_id][:-tail] if tail > 0 else pre[cat_id]
    valid_scores = sc[cat_id][:-tail] if tail > 0 else sc[cat_id]
    
    # 配列の長さを最小の長さに揃える
    min_length = min(len(valid_recall), len(valid_precision), len(valid_scores))
    valid_recall = valid_recall[:min_length]
    valid_precision = valid_precision[:min_length]
    valid_scores = valid_scores[:min_length]
    
    if len(valid_recall) == 0 or len(valid_precision) == 0:
        print(f"カテゴリ '{cat_name[cat_id]}' には有効なデータがありません。")
        continue
    
    # AP計算とPR曲線の補間
    ap, mrec, mpre = voc_ap(valid_recall.copy(), valid_precision.copy())
    
    # 個別のPR曲線を描画
    if cat_id < len(axes):
        axes[cat_id].step(mrec[1:-1], mpre[1:-1], where='post', linewidth=2)
        axes[cat_id].set_xlabel('Recall')
        axes[cat_id].set_ylabel('Precision')
        axes[cat_id].set_title(f'{cat_name[cat_id]}\nAP = {ap*100:.2f}%')
        axes[cat_id].grid(True)
        axes[cat_id].set_xlim([0.0, 1.0])
        axes[cat_id].set_ylim([0.0, 1.0])
    
    # 全カテゴリまとめたPR曲線に追加
    plt.step(mrec[1:-1], mpre[1:-1], where='post', linewidth=2, 
             label=f'{cat_name[cat_id]} (AP={ap*100:.2f}%)')
    
    # 個別のPR曲線を保存
    plt.figure(figsize=(8, 6))
    plt.step(mrec[1:-1], mpre[1:-1], where='post', linewidth=2, 
             label=f'AP = {ap*100:.2f}%')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {cat_name[cat_id]}')
    plt.legend()
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.savefig(f"{pr_curve_output_dir}pr_curve_{cat_name[cat_id].replace('/', '_')}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # データをCSVとして保存
    try:
        import pandas as pd
        df = pd.DataFrame({
            'recall': valid_recall,
            'precision': valid_precision,
            'scores': valid_scores
        })
        df = df.sort_values(by='scores', ascending=False).reset_index(drop=True)
        df.to_csv(f"{pr_curve_output_dir}pr_data_{cat_name[cat_id].replace('/', '_')}.csv", 
                  index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"カテゴリ '{cat_name[cat_id]}' のCSV保存でエラーが発生しました: {e}")
        # CSVなしでも続行

# 最後のサブプロットを非表示にする（5カテゴリの場合）
if len(axes) > cat_len:
    axes[-1].set_visible(False)

# 個別PR曲線のサブプロット全体を保存
plt.figure(fig.number)
plt.tight_layout()
plt.savefig(f"{pr_curve_output_dir}all_pr_curves_subplots.png", dpi=300, bbox_inches='tight')
plt.close()

# 全カテゴリをまとめたPR曲線を保存
plt.figure(figsize=(10, 8))
for cat_id in range(cat_len):
    tail = pre[cat_id].count(-1)
    valid_recall = rec[cat_id][:-tail] if tail > 0 else rec[cat_id]
    valid_precision = pre[cat_id][:-tail] if tail > 0 else pre[cat_id]
    
    if len(valid_recall) == 0 or len(valid_precision) == 0:
        continue
        
    ap, mrec, mpre = voc_ap(valid_recall.copy(), valid_precision.copy())
    plt.step(mrec[1:-1], mpre[1:-1], where='post', linewidth=2, 
             label=f'{cat_name[cat_id]} (AP={ap*100:.2f}%)')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves for All Categories')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.savefig(f"{pr_curve_output_dir}all_categories_pr_curves.png", 
            dpi=300, bbox_inches='tight')
plt.close()

# mAP計算と表示
all_aps = []
print("\n--- Average Precision (AP) Results ---")
for cat_id in range(cat_len):
    tail = pre[cat_id].count(-1)
    valid_recall = rec[cat_id][:-tail] if tail > 0 else rec[cat_id]
    valid_precision = pre[cat_id][:-tail] if tail > 0 else pre[cat_id]
    
    if len(valid_recall) > 0 and len(valid_precision) > 0:
        ap, _, _ = voc_ap(valid_recall.copy(), valid_precision.copy())
        all_aps.append(ap)
        print(f"{cat_name[cat_id]}: AP = {ap*100:.2f}%")
    else:
        print(f"{cat_name[cat_id]}: No valid data")

if all_aps:
    mAP = np.mean(all_aps)
    print(f"\n--- Mean Average Precision (mAP) ---")
    print(f"mAP = {mAP*100:.2f}%")

print(f"\nPR曲線は {pr_curve_output_dir} に保存されました。")