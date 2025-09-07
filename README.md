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
