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

