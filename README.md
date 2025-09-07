# conda-activate-odise

conda activate odise


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

