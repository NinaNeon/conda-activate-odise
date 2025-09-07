(odise) nina@DESKTOP-QF7UBMR:/mnt/c/Users/USER/Desktop/ODISE$ conda activate odise

# 目標檔案
P=/home/nina/Mask2Former/mask2former/data/datasets/register_coco_panoptic_annos_semseg.py

# 備份
cp "$P" "${P}.bak_force"

# 1) 在檔首插入 _safe_del 定義（保證存在）
{
  printf 'def _safe_del(meta, attr):\n    if hasattr(meta, attr):\n        delattr(meta, attr)\n\n'
  cat "$P"
} > "${P}.patched" && mv "${P}.patched" "$P"

# 2) 將所有 delattr(...) 換成 _safe_del(...)
sed -i 's/delattr(MetadataCatalog.get(panoptic_name), "\([^"]*\)")/_safe_del(MetadataCatalog.get(panoptic_name), "\1")/g' "$P"

# 3) 驗證檔案前幾行 & 出現幾次 _safe_del
head -n 12 "$P"
grep -n "_safe_del(" "$P" | head
def _safe_del(meta, attr):
    if hasattr(meta, attr):
        delattr(meta, attr)

# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.utils.file_io import PathManager
1:def _safe_del(meta, attr):
137:    _safe_del(MetadataCatalog.get(panoptic_name), "thing_classes")
138:    _safe_del(MetadataCatalog.get(panoptic_name), "thing_colors")
(odise) nina@DESKTOP-QF7UBMR:/mnt/c/Users/USER/Desktop/ODISE$ # 保持既有路徑與動態庫
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
TORCH_LIB_DIR=$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__),"lib"))')
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:$LD_LIBRARY_PATH"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="/home/nina/Mask2Former:/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops:$PYTHONPATH"

cd /mnt/c/Users/USER/Desktop/ODISE
python demo/demo.py \
  --input demo/examples/coco.jpg \
  --output demo/coco_pred.jpg \
  --vocab "black pickup truck, pickup truck; blue sky, sky"




(odise) nina@DESKTOP-QF7UBMR:/mnt/c/Users/USER/Desktop/ODISE$ conda activate odise
export CUDA_VISIBLE_DEVICES=""      # 強制只用 CPU
cd /mnt/c/Users/USER/Desktop/ODISE

python -u demo/demo.py \
  --input demo/examples/coco.jpg \
  --output demo/coco_pred.jpg \
  --vocab "black pickup truck, pickup truck; blue sky, sky"
/home/nina/miniconda3/envs/odise/lib/python3.9/site-packages/detectron2/config/lazy.py:153: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  return old_import(name, globals, locals, fromlist=fromlist, level=level)
/home/nina/miniconda3/envs/odise/lib/python3.9/site-packages/pytorch_lightning/utilities/distributed.py:258: LightningDeprecationWarning: `pytorch_lightning.utilities.distributed.rank_zero_only` has been deprecated in v1.8.1 and will be removed in v2.0.0. You can import it from `pytorch_lightning.utilities` instead.
  rank_zero_deprecation(
[09/08 00:43:30 odise]: extra classes: [['black pickup truck', 'pickup truck'], ['blue sky', 'sky']]
[09/08 00:43:30 d2.data.dataset_mapper]: [DatasetMapper] Augmentations used in inference: [ResizeShortestEdge(short_edge_length=(1024, 1024), max_size=2560, sample_style='choice')]
LatentDiffusion: Running in eps-prediction mode
DiffusionWrapper has 859.52 M params.
making attention of type 'vanilla' with 512 in_channels
Working with z of shape (1, 4, 32, 32) = 4096 dimensions.
making attention of type 'vanilla' with 512 in_channels
/home/nina/miniconda3/envs/odise/lib/python3.9/site-packages/huggingface_hub/file_download.py:945: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
The checkpoint state_dict contains keys that are not used by the model:
  model_ema.{decay, num_updates}
  cond_stage_model.transformer.text_model.embeddings.position_ids
/home/nina/miniconda3/envs/odise/lib/python3.9/site-packages/open_clip/factory.py:388: UserWarning: These pretrained weights were trained with QuickGELU activation but the model config does not have that enabled. Consider using a model config with a "-quickgelu" suffix or enable with a flag.
  warnings.warn(
[09/08 00:43:46 odise.modeling.backbone.feature_extractor]: backbone_in_size: (512, 512), slide_training: True,
slide_inference: True,
min_stride: 4, max_stride: 32,
projection_dim: 512,
out_feature_channels: {'s2': 512, 's3': 512, 's4': 512, 's5': 512}
out_feature_strides: {'s2': 4, 's3': 8, 's4': 16, 's5': 32}
use_checkpoint: True

[09/08 00:43:48 odise.modeling.meta_arch.clip]: Using provided CLIP model
[09/08 00:43:57 odise.modeling.meta_arch.clip]: Built text_embed of shape torch.Size([254, 768]) for 133 labels: (('person', 'child', 'girl', 'boy', 'woman', 'man', 'people', 'childeren', 'girls', 'boys', 'women', 'men', 'lady', 'guy', 'ladies', 'guys', 'clothes'), ('bicycle', 'bicycles', 'bike', 'bikes'), ('car', 'cars'), ('motorcycle', 'motorcycles'), ('airplane', 'airplanes'), ('bus', 'buses'), ('train', 'trains', 'locomotive', 'locomotives', 'freight train'), ('truck', 'trucks'), ('boat', 'boats'), ('traffic light',), ('fire hydrant',), ('stop sign',), ('parking meter',), ('bench', 'benches'), ('bird', 'birds'), ('cat', 'cats', 'kitties', 'kitty'), ('dog', 'dogs', 'puppy', 'puppies'), ('horse', 'horses', 'foal'), ('sheep',), ('cow', 'cows', 'calf'), ('elephant', 'elephants'), ('bear', 'bears'), ('zebra', 'zebras'), ('giraffe', 'giraffes'), ('backpack', 'backpacks'), ('umbrella', 'umbrellas'), ('handbag', 'handbags'), ('tie',), ('suitcase', 'suitcases'), ('frisbee',), ('skis',), ('snowboard',), ('sports ball',), ('kite', 'kites'), ('baseball bat',), ('baseball glove',), ('skateboard',), ('surfboard',), ('tennis racket',), ('bottle', 'bottles', 'water bottle'), ('wine glass', 'wine glasses', 'wineglass'), ('cup', 'cups', 'water cup', 'water glass'), ('fork', 'forks'), ('knife', 'knives'), ('spoon', 'spoons'), ('bowl', 'bowls'), ('banana', 'bananas'), ('apple', 'apples', 'apple fruit'), ('sandwich', 'sandwiches'), ('orange fruit',), ('broccoli',), ('carrot', 'carrots'), ('hot dog',), ('pizza',), ('donut', 'donuts'), ('cake', 'cakes'), ('chair', 'chairs'), ('couch', 'sofa', 'sofas'), ('potted plant', 'potted plants', 'pottedplant', 'pottedplants', 'planter', 'planters'), ('bed', 'beds'), ('dining table', 'dining tables', 'diningtable', 'diningtables', 'plate', 'plates', 'diningtable tablecloth'), ('toilet',), ('tv',), ('laptop',), ('mouse',), ('tv remote', 'remote control'), ('keyboard',), ('cell phone', 'mobile'), ('microwave',), ('oven', 'ovens'), ('toaster',), ('sink', 'sinks'), ('refrigerator', 'fridge'), ('book', 'books'), ('clock',), ('vase', 'vases'), ('scissor', 'scissors'), ('teddy bear', 'teddy bears'), ('hair drier',), ('toothbrush', 'toothbrushes'), ('banner', 'banners'), ('blanket', 'blankets'), ('bridge',), ('cardboard',), ('counter',), ('curtain', 'curtains'), ('door', 'doors'), ('wood floor',), ('flower', 'flowers'), ('fruit', 'fruits'), ('gravel',), ('house',), ('lamp', 'bulb', 'lamps', 'bulbs'), ('mirror',), ('tennis net',), ('pillow', 'pillows'), ('platform',), ('playingfield', 'tennis court', 'baseball feild', 'soccer field', 'tennis field'), ('railroad',), ('river',), ('road',), ('roof',), ('sand',), ('sea', 'sea wave', 'wave', 'waves'), ('shelf',), ('snow',), ('stairs',), ('tent',), ('towel',), ('brick wall',), ('stone wall',), ('tile wall',), ('wood wall',), ('water',), ('window blind',), ('window',), ('tree', 'trees', 'palm tree', 'bushes'), ('fence', 'fences'), ('ceiling',), ('sky', 'clouds'), ('cabinet', 'cabinets'), ('table',), ('floor', 'flooring', 'tile floor'), ('pavement',), ('mountain', 'mountains'), ('grass',), ('dirt',), ('paper',), ('food',), ('building', 'buildings'), ('rock',), ('wall', 'walls'), ('rug',))
Traceback (most recent call last):
  File "/mnt/c/Users/USER/Desktop/ODISE/demo/demo.py", line 385, in <module>
    model.to(cfg.train.device)
  File "/home/nina/miniconda3/envs/odise/lib/python3.9/site-packages/torch/nn/modules/module.py", line 989, in to
    return self._apply(convert)
  File "/home/nina/miniconda3/envs/odise/lib/python3.9/site-packages/torch/nn/modules/module.py", line 641, in _apply
    module._apply(fn)
  File "/home/nina/miniconda3/envs/odise/lib/python3.9/site-packages/torch/nn/modules/module.py", line 641, in _apply
    module._apply(fn)
  File "/home/nina/miniconda3/envs/odise/lib/python3.9/site-packages/torch/nn/modules/module.py", line 641, in _apply
    module._apply(fn)
  [Previous line repeated 6 more times]
  File "/home/nina/miniconda3/envs/odise/lib/python3.9/site-packages/torch/nn/modules/module.py", line 664, in _apply
    param_applied = fn(param)
  File "/home/nina/miniconda3/envs/odise/lib/python3.9/site-packages/torch/nn/modules/module.py", line 987, in convert
    return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
  File "/home/nina/miniconda3/envs/odise/lib/python3.9/site-packages/torch/cuda/__init__.py", line 229, in _lazy_init
    torch._C._cuda_init()
RuntimeError: No CUDA GPUs are available
(odise) nina@DESKTOP-QF7UBMR:/mnt/c/Users/USER/Desktop/ODISE$
