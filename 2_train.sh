#!/bin/bash
set -e # <-- Thoát ngay nếu có lỗi
ENV_NAME="slotdiff_final"

echo "============================================================="
echo " [BUOC 1/3] KICH HOAT MOI TRUONG '$ENV_NAME'..."
echo "============================================================="
echo ""

# 1. Kích hoạt môi trường
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
echo "Da kich hoat moi truong: $CONDA_DEFAULT_ENV"
echo ""

echo "============================================================="
echo " [BUOC 2/3] CAI DAT / CAP NHAT THU VIEN (PIP)"
echo "============================================================="
echo ""
echo "Bat dau cai dat PyTorch 12.1 va cac thu vien..."
echo "(Buoc nay co the mat vai phut...)"

# 2. Chạy PIP (Dùng file requirements_linux.txt)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

echo "Cai dat PIP thanh cong."
echo ""

echo "============================================================="
echo " [BUOC 3/3] BAT DAU TRAIN 5000 EPOCH..."
echo "============================================================="
echo ""

# 3. Chạy train (dùng '\' để xuống dòng)
python train.py \
    --problem=Train_ALL \
    --model_type=MTL \
    --epochs=5000 \
    --enable_slot_diffusion \
    --checkpoint="./pretrained/SlotDIff/epoch-500.pt"


python train.py \
    --problem=Train_ALL \
    --model_type=MOE_Mixed \
    --epochs=5000 \ 
    --checkpoint="./pretrained/MixedMOE/epoch-500.pt"


echo ""
echo "============================================================="
echo " HOAN TAT 5000 EPOCHS CHO 2 MODEL! DA LUU CHECKPOINT."
echo "============================================================="