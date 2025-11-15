#!/bin/bash
set -e  # <-- Thoát ngay nếu có lỗi
ENV_NAME="slotdiff_final"
PYTHON_VER="3.10"

echo "============================================================="
echo " [BUOC 1/1] TAO MOI TRUONG CONDA '$ENV_NAME'"
echo "============================================================="
echo ""
echo "Bat dau tao moi truong '$ENV_NAME' voi Python=$PYTHON_VER..."

# 1. Tạo môi trường
conda create -n "$ENV_NAME" python="$PYTHON_VER" -y

echo ""
echo "============================================================="
echo " DA TAO XONG MOI TRUONG."
echo " Chay file '2_train.sh' de kich hoat va train."
echo "============================================================="