
`MultiScaleDeformableAttention` 又找不到是因為你這個新 shell 沒把 **PYTHONPATH / LD\_LIBRARY\_PATH** 設回來（之前那次能過就是因為有設）。照下面一次跑完就好：

```bash
conda activate odise

# 0) 只用 CPU
export CUDA_VISIBLE_DEVICES=""

# 1) 讓 Python 找得到 MSDA .so 與 Mask2Former 的套件
export PYTHONPATH="/home/nina/Mask2Former:/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops:$PYTHONPATH"

# 2) 讓動態連結器找得到 PyTorch 的 libc10 等
TORCH_LIB_DIR=$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__),"lib"))')
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$TORCH_LIB_DIR:$LD_LIBRARY_PATH"

# 3) 快測：MSDA 能 import、CUDA 已關
python - <<'PY'
import os, sys, torch
print("CUDA_VISIBLE_DEVICES =", repr(os.environ.get("CUDA_VISIBLE_DEVICES")))
print("torch.cuda.is_available() ->", torch.cuda.is_available())
print("sys.path[0:3] ->", sys.path[0:3])
import MultiScaleDeformableAttention as MSDA
print("MSDA OK:", MSDA.__file__)
PY

# 4) 跑 demo（CPU）
cd /mnt/c/Users/USER/Desktop/ODISE
python -u demo/demo.py \
  --input demo/examples/coco.jpg \
  --output demo/coco_pred.jpg \
  --vocab "black pickup truck, pickup truck; blue sky, sky" \
  --opts train.device=cpu model.device=cpu
```


```bash
conda activate odise

# 1) 移除之前裝到的空殼
pip uninstall -y latent-diffusion

# 2) 裝 unzip（解壓 zip 需要）
sudo apt-get update -y && sudo apt-get install -y unzip

# 3) 下載並安裝 stable-diffusion（內含 ldm 套件）
tmp=/tmp/ldm && rm -rf "$tmp" && mkdir -p "$tmp" && cd "$tmp"
wget -O sd.zip https://codeload.github.com/CompVis/stable-diffusion/zip/refs/heads/main
unzip sd.zip
cd stable-diffusion-main
pip install --no-deps -U .

# 4) 驗證 ldm 是否可匯入
python - <<'PY'
import ldm, importlib
m = importlib.import_module("ldm.models.diffusion.ddpm")
print("ldm OK ->", m.__name__)
PY
```


```bash
conda activate odise

# 1) 確認你有剛剛抓下來的 stable-diffusion 目錄
SD=/tmp/ldm/stable-diffusion-main
ls -d "$SD/ldm" || echo "!! 找不到 $SD/ldm，若不存在先重新下載 zip 再解壓"

# 2) 臨時加到這個 shell 的 PYTHONPATH
export PYTHONPATH="$SD:$PYTHONPATH"

# 3) 快測 import
python - <<'PY'
import sys
print("sys.path[0:2] =", sys.path[:2])
import ldm, importlib
m = importlib.import_module("ldm.models.diffusion.ddpm")
print("ldm OK ->", m.__name__)
PY

# 4) 跑 demo（可先用 CPU 驗證流程通不通）
# export CUDA_VISIBLE_DEVICES=""
cd /mnt/c/Users/USER/Desktop/ODISE
python -u demo/demo.py \
  --input demo/examples/coco.jpg \
  --output demo/coco_pred.jpg \
  --vocab "black pickup truck, pickup truck; blue sky, sky"

```



# conda-activate-odise
<img width="784" height="598" alt="image" src="https://github.com/user-attachments/assets/5fe5e7b7-228c-495d-81c5-4bc746d2dffc" />

conda activate odise

cd /mnt/c/Users/USER/Desktop/ODISE

python -u demo/demo.py \
  --input demo/examples/coco.jpg \
  --output demo/coco_pred.jpg \
  --vocab "black pickup truck, pickup truck; blue sky, sky"

https://chatgpt.com/share/68bdb33c-b724-8012-94a4-3b4adbdcd503

`MultiScaleDeformableAttention` 又找不到是因為你這個新 shell 沒把 **PYTHONPATH / LD\_LIBRARY\_PATH** 設回來（之前那次能過就是因為有設）。照下面一次跑完就好：

```bash
conda activate odise

# 0) 只用 CPU
export CUDA_VISIBLE_DEVICES=""

# 1) 讓 Python 找得到 MSDA .so 與 Mask2Former 的套件
export PYTHONPATH="/home/nina/Mask2Former:/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops:$PYTHONPATH"

# 2) 讓動態連結器找得到 PyTorch 的 libc10 等
TORCH_LIB_DIR=$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__),"lib"))')
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$TORCH_LIB_DIR:$LD_LIBRARY_PATH"

# 3) 快測：MSDA 能 import、CUDA 已關
python - <<'PY'
import os, sys, torch
print("CUDA_VISIBLE_DEVICES =", repr(os.environ.get("CUDA_VISIBLE_DEVICES")))
print("torch.cuda.is_available() ->", torch.cuda.is_available())
print("sys.path[0:3] ->", sys.path[0:3])
import MultiScaleDeformableAttention as MSDA
print("MSDA OK:", MSDA.__file__)
PY

# 4) 跑 demo（CPU）
cd /mnt/c/Users/USER/Desktop/ODISE
python -u demo/demo.py \
  --input demo/examples/coco.jpg \
  --output demo/coco_pred.jpg \
  --vocab "black pickup truck, pickup truck; blue sky, sky" \
  --opts train.device=cpu model.device=cpu
```


# 1) 基本建置工具
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build \
    libglib2.0-0 libsm6 libxext6 libxrender-dev

# 2) 安裝 CUDA 11.6 Toolkit（提供 nvcc）
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-6

# 3) 設定環境變數（本次 shell；可寫入 ~/.bashrc）
export CUDA_HOME=/usr/local/cuda-11.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 4) 檢查 nvcc
nvcc -V   # 應顯示 release 11.6

# 5) 安裝 GPU 版 detectron2（對 torch1.13.1 + cu116）
conda activate odise
pip uninstall -y detectron2 || true
pip install "git+https://github.com/facebookresearch/detectron2.git@v0.6"

# 6) 驗證
python -c "import torch, detectron2; print('torch', torch.__version__, 'CUDA?', torch.cuda.is_available())"

# 7) 跑 demo（會用 GPU）
cd /mnt/c/Users/USER/Desktop/ODISE
python demo/demo.py \
  --input demo/examples/coco.jpg \
  --output demo/coco_pred.jpg \
  --vocab "black pickup truck, pickup truck; blue sky, sky"


  conda activate odise

# 1) 釘住會出事的版本：NumPy 回到 <2、Pillow 回到 <10
pip install "numpy<2" "pillow<10" --upgrade

# 2) 釘回與 torch=1.13.1 相容的 torchvision
pip install "torchvision==0.14.1" --no-deps

# 3) 升級 conda 的 libstdc++/libgcc，解決 GLIBCXX_3.4.32
conda install -y -c conda-forge "libstdcxx-ng>=12" "libgcc-ng>=12"

# 確保 conda 的 lib 會被優先載入
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# 4) 重新安裝 detectron2（先走 CPU 版把 demo 跑通；之後要 GPU 我再幫你切回）
pip uninstall -y detectron2 || true
export FORCE_CUDA=0
pip install --no-cache-dir "git+https://github.com/facebookresearch/detectron2.git@v0.6"

# 5) 快速自檢
python - <<'PY'
import torch, PIL, numpy as np
print("Torch:", torch.__version__, "CUDA?", torch.cuda.is_available())
print("Pillow:", PIL.__version__)
print("NumPy:", np.__version__)
PY
# 安裝 CUDA 11.6 Toolkit（提供 nvcc），之前若已裝可略過
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build \
    libglib2.0-0 libsm6 libxext6 libxrender-dev
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-6

# 設定
export CUDA_HOME=/usr/local/cuda-11.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
nvcc -V   # 應顯示 release 11.6

# 重裝 GPU 版 detectron2（移除 FORCE_CUDA）
conda activate odise
pip uninstall -y detectron2
pip install --no-cache-dir "git+https://github.com/facebookresearch/detectron2.git@v0.6"

python - <<'PY'
import torch, detectron2
print("Torch:", torch.__version__, "CUDA?", torch.cuda.is_available())
print("D2:", detectron2.__version__)
PY

