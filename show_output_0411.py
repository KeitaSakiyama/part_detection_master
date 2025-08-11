# 作業ディレクトリをGLIPにする
import os
os.chdir('./GLIP')

# make_dataset.pyで作成したデータセットの情報が記載されているtestファイルを指定
task_config = "../../bridge_data/configs/20250411共有_test.yaml"

dir_name = "20250411共有_valid"
ft_type = "full"
shot_num = "all"
test_seed = "1"

output_path = "../"

# 推論に使用するモデルの重みと情報ファイルを指定
config_file = output_path + "/1st_OUTPUT/" + dir_name + "/" + ft_type + "/" + str(shot_num) + "_shot/ft_task_" + test_seed + "/config.yml"
weight_file = output_path + "/1st_OUTPUT/" + dir_name + "/" + ft_type + "/" + str(shot_num) + "_shot/ft_task_" + test_seed + "/model_best.pth"

# カテゴリを指定
ctg_list = ['wire strand', 'cable clamp', 'cable-clamp bolt', 'turnbuckle', 'tower saddle', \
            'anchorage piece', 'wire clip', 'rod anchorage nut', 'anchorage pin', 'rubber boot']


# # ストランドロープを4種類の錆異常つきで検出させる場合（テスト用のモデルも共有）以下をコメントアウト
# task_config = "../../bridge_data/configs/cable_with_sabi_test.yaml"
# config_file = "./MODEL/config_cable_with_sabi.yml"
# weight_file = "./MODEL/model_cable_with_sabi.pth"
# # カテゴリを指定
# ctg_list = ["wire strand with full red rust", "wire strand with half red rust", "wire strand with half red rust", "wire strand with white rust"]

# # 10種類の部品で検出させる場合（テスト用のモデルも共有）以下をコメントアウト
task_config = "../../bridge_data/configs/10parts_test.yaml"
config_file = "./MODEL/config_10parts.yml"
weight_file = "./MODEL/model_10parts.pth"
# カテゴリを指定
ctg_list = ['wire strand', 'cable clamp', 'cable-clamp bolt', 'turnbuckle', 'tower saddle', 'anchorage piece', 'wire clip', 'rod anchorage nut', 'anchorage pin', 'rubber boot']


# 検出の出力を表示する際のスコアの閾値
out_thresh = 0.5

# 検出させたい画像が入っているフォルダを指定
folder_path = "../../bridge_data/20250411共有/01_採用ファイル/"

# 検出結果の画像を保存するフォルダ
output_image_dir = "../test_output/20250411共有/"

# # 検出に使用したい画像のファイル名のリスト
# image_list = ["a0021_IMG_9635.JPG", "a0033_JUUX3070.JPG", "a0068_P8260420.JPG", "a0117_IMG_0394.JPG", "c0128_P5312919.JPG", "b0001_BHXH1282.JPG"]

# もしフォルダの画像すべてに対して推論させたいときは以下をコメントアウトしてください
# 画像の拡張子リスト
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
# 指定したフォルダの全画像ファイル名のリストを取得
# image_list = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1] in IMAGE_EXTENSIONS]
image_list = [f for f in os.listdir(folder_path)
              if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
              and not f.startswith("._")]
print("image_list:", image_list)

# 必要なパッケージのimport（前準備）
import sys
import time
from tqdm import tqdm
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 20, 12
import json
import cv2
import csv
import ast
import torch
from torch.utils.cpp_extension import CUDA_HOME
print(torch.cuda.is_available(), CUDA_HOME)

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from maskrcnn_benchmark.engine.predictor_glip import *
from maskrcnn_benchmark.modeling.detector.generalized_vl_rcnn import *

cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

class args:
    config_file = config_file
    weight = weight_file
    task_config = task_config
    local_rank = 0
    world_size = 1
    dist_url = "env://"
    opts = [
        "TEST.IMS_PER_BATCH", 2,
        "SOLVER.IMS_PER_BATCH", 2,
        "TEST.EVAL_TASK", "detection",
        "DATASETS.TRAIN_DATASETNAME_SUFFIX", "_grounding",
        "DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE", False,
        "DATASETS.USE_OVERRIDE_CATEGORY", True,
        "DATASETS.USE_CAPTION_PROMPT", False #tomiya
    ]
cfg_ = cfg.clone()
cfg_.defrost()
cfg_.merge_from_file(task_config)
cfg_.merge_from_list(args.opts)

class GLIPDemo(object):
    def __init__(self,
                 cfg,
                 confidence_threshold=0.7,
                 min_image_size=None,
                 show_mask_heatmaps=False,
                 masks_per_dim=5,
                 ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        
        self.min_image_size = min_image_size
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        # used to make colors for each tokens
        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold

        self.tokenizer = self.build_tokenizer()
        self.model.rpn.box_selector_test.pre_nms_thresh=.0001 #here

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size) if self.min_image_size is not None else lambda x: x,
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def build_tokenizer(self):
        cfg = self.cfg
        tokenizer = None
        if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "bert-base-uncased":
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        elif cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
            from transformers import CLIPTokenizerFast
            if cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
                tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                              from_slow=True, mask_token='ðŁĴĳ</w>')
            else:
                tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                              from_slow=True)
        return tokenizer

    def run_ner(self, caption):
        noun_phrases = find_noun_phrases(caption)
        noun_phrases = [remove_punctuation(phrase) for phrase in noun_phrases]
        noun_phrases = [phrase for phrase in noun_phrases if phrase != '']
        relevant_phrases = noun_phrases
        labels = noun_phrases
        self.entities = labels

        tokens_positive = []

        for entity, label in zip(relevant_phrases, labels):
            try:
                # search all occurrences and mark them as different entities
                for m in re.finditer(entity, caption.lower()):
                    tokens_positive.append([[m.start(), m.end()]])
            except:
                print("noun entities:", noun_phrases)
                print("entity:", entity)
                print("caption:", caption.lower())

        return tokens_positive

    def inference(self, original_image, original_caption):
        predictions = self.compute_prediction(original_image, original_caption)
        top_predictions = self._post_process_fixed_thresh(predictions)
        return top_predictions

    def run_on_web_image(self, original_image, original_caption, thresh=0.5):
        predictions = self.compute_prediction(original_image, original_caption)
        top_predictions = self._post_process(predictions, thresh)

        result = original_image.copy()
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)
        result = self.overlay_boxes(result, top_predictions)
        result = self.overlay_entity_names(result, top_predictions)
        if self.cfg.MODEL.MASK_ON:
            result = self.overlay_mask(result, top_predictions)

        return result, top_predictions

    def compute_prediction(self, original_image, original_caption):
        # image
        image = self.transforms(original_image)
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # caption
        tokenized = self.tokenizer([original_caption], return_tensors="pt")
        tokens_positive = self.run_ner(original_caption)
        print(tokenized)
        print(tokens_positive)
        # process positive map
        positive_map = create_positive_map(tokenized, tokens_positive)

        if self.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
            plus = 1
        else:
            plus = 0

        positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map, plus=plus)
        self.plus = plus
        self.positive_map_label_to_token = positive_map_label_to_token
        print(positive_map_label_to_token)
        tic = timeit.time.perf_counter()

        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list, captions=[original_caption], positive_map=positive_map_label_to_token)
            predictions = [o.to(self.cpu_device) for o in predictions]
        #print("inference time per image: {}".format(timeit.time.perf_counter() - tic))

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)

        return prediction
    
    def compute_prediction_new(self, original_image, original_caption,tokenized,tokens_positive,entities):
        # image
        self.entities = entities
        image = self.transforms(original_image)
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # caption
        # tokenized = self.tokenizer([original_caption], return_tensors="pt")
        # tokens_positive = self.run_ner(original_caption)
        # process positive map
        positive_map = create_positive_map(tokenized, tokens_positive)

        if self.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
            plus = 1
        else:
            plus = 0

        positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map, plus=plus)
        self.plus = plus
        self.positive_map_label_to_token = positive_map_label_to_token
        tic = timeit.time.perf_counter()

        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list, captions=[original_caption], positive_map=positive_map_label_to_token)
            predictions = [o.to(self.cpu_device) for o in predictions]
        #print("inference time per image: {}".format(timeit.time.perf_counter() - tic))

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            with open(json_path) as f:
                json_data = json.load(f)
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)

        return prediction


    def _post_process_fixed_thresh(self, predictions):
        scores = predictions.get_field("scores")
        labels = predictions.get_field("labels").tolist()
        thresh = scores.clone()
        for i, lb in enumerate(labels):
            if isinstance(self.confidence_threshold, float):
                thresh[i] = self.confidence_threshold
            elif len(self.confidence_threshold) == 1:
                thresh[i] = self.confidence_threshold[0]
            else:
                thresh[i] = self.confidence_threshold[lb - 1]
        keep = torch.nonzero(scores > thresh).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def _post_process(self, predictions, threshold=0.5):
        scores = predictions.get_field("scores")
        labels = predictions.get_field("labels").tolist()
        thresh = scores.clone()
        for i, lb in enumerate(labels):
            if isinstance(self.confidence_threshold, float):
                thresh[i] = threshold
            elif len(self.confidence_threshold) == 1:
                thresh[i] = threshold
            else:
                thresh[i] = self.confidence_threshold[lb - 1]
        keep = torch.nonzero(scores > thresh).squeeze(1)
        predictions = predictions[keep]

        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = (30 * (labels[:, None] - 1) + 1) * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 2)

        return image

    def overlay_scores(self, image, predictions):
        scores = predictions.get_field("scores")
        boxes = predictions.bbox

        for box, score in zip(boxes, scores):
            box = box.to(torch.int64)
            image = cv2.putText(image, '%.3f' % score,
                                (int(box[0]), int((box[1] + box[3]) / 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                (255, 255, 255), 1)

        return image

    def overlay_entity_names(self, image, predictions, names=None):
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        new_labels = []
        if self.entities and self.plus:
            for i in labels:
                if i <= len(self.entities):
                    new_labels.append(self.entities[i - self.plus])
                else:
                    new_labels.append('object')
            # labels = [self.entities[i - self.plus] for i in labels ]
        else:
            new_labels = ['object' for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, new_labels):
            x, y = box[:2]
            s = template.format(label, score)
            cv2.putText(
                image, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
            )

        return image

    def overlay_mask(self, image, predictions):
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")

        colors = self.compute_colors_for_labels(labels).tolist()

        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None].astype(np.uint8)
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.drawContours(image, contours, -1, color, 2)

        composite = image

        return composite

    def create_mask_montage(self, image, predictions):
        masks = predictions.get_field("mask")
        masks_per_dim = self.masks_per_dim
        masks = L.interpolate(
            masks.float(), scale_factor=1 / masks_per_dim
        ).byte()
        height, width = masks.shape[-2:]
        max_masks = masks_per_dim ** 2
        masks = masks[:max_masks]
        # handle case where we have less detections than max_masks
        if len(masks) < max_masks:
            masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
            masks_padded[: len(masks)] = masks
            masks = masks_padded
        masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
        result = torch.zeros(
            (masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
        )
        for y in range(masks_per_dim):
            start_y = y * height
            end_y = (y + 1) * height
            for x in range(masks_per_dim):
                start_x = x * width
                end_x = (x + 1) * width
                result[start_y:end_y, start_x:end_x] = masks[y, x]

        return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET), None

glip_demo = GLIPDemo(
    cfg_,
    min_image_size=800,
    confidence_threshold=0.5,
    show_mask_heatmaps=False
)

# GPUの数を設定している（基本 1）
num_gpus = 1
distributed = False

glip_demo.model.rpn.box_selector_test.pre_nms_thresh = .0001
glip_demo.color = 255

def run_on_web_image_new(self, original_image, ctg_list, thresh=0.5):
    
    entities = ctg_list
    original_caption = ". ".join(ctg_list)+"."
    
    tokenized = glip_demo.tokenizer([original_caption], return_tensors="pt")

    # entitesをつないだ何文字目にあるかのリスト（最初の文字は0文字目，つなげるときは各カテゴリの間に". "が入る）
    tokens_positive = []
    start_idx = 0
    for word in ctg_list:
        end_idx = start_idx + len(word)
        tokens_positive.append([[start_idx, end_idx]])
        start_idx = end_idx + 2

    predictions = self.compute_prediction_new(original_image, original_caption, tokenized,tokens_positive,entities)

    top_predictions = self._post_process(predictions, thresh)

    result = original_image.copy()
    if self.show_mask_heatmaps:
        return self.create_mask_montage(result, top_predictions)
    result = self.overlay_boxes(result, top_predictions)
    result = self.overlay_entity_names(result, top_predictions)
    if self.cfg.MODEL.MASK_ON:
        result = self.overlay_mask(result, top_predictions)

    return result, top_predictions


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

for file_name in image_list:
    each_image = cv2.imread(folder_path + file_name)
    _, top_predictions = run_on_web_image_new(glip_demo, each_image, ctg_list, out_thresh)
    image_rgb = cv2.cvtColor(each_image, cv2.COLOR_BGR2RGB)
    bb_list = top_predictions.bbox.tolist()
    ct_list = [ctg_list[int(tmp)-1] for tmp in top_predictions.get_field("labels").tolist()]
    sc_list = top_predictions.get_field("scores").tolist()
    
    bb_list = bb_list[::-1]
    ct_list = ct_list[::-1]
    sc_list = sc_list[::-1]

    image_height = each_image.shape[0]
    image_width = each_image.shape[1]
    figsize_input=(image_width/100, image_height/100)
    plot_results(image_rgb, bb_list, ct_list, sc_list, figsize_input)
    plt.savefig(output_image_dir + "/" + file_name.split(".")[0] + "_output.jpg", bbox_inches='tight', pad_inches=0.1, dpi=100)
