#!/usr/bin/env bash
# AISD / Cl-SegNet 分割复现一键流程（速查）。
# 运行前请先：
#   1) 按 reproduce/README.md 第 1 节安装环境；
#   2) 编辑 nnUNet/nnunet/paths.py 的三个路径；
#   3) 修改下面的占位路径变量。
set -e

# ====== 改成你的实际路径 ======
DICOM_ROOT=/path/to/aisd/dicom
MASK_ROOT=/path/to/aisd/mask
AISD_NII=/path/to/AISD_nii
NNUNET_PREPROCESSED=/your/path/nnUNet_preprocessed
GPU=0
# =============================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1. 安装（首次运行取消注释）
# (cd "$REPO_ROOT/nnUNet" && pip install -e .)
# (cd "$REPO_ROOT/ClSeg_package" && pip install -e .)

# 2. DICOM + PNG 掩膜 -> 对齐的 nii.gz
python "$REPO_ROOT/reproduce/aisd_dicom_to_nii.py" \
    --dicom-root "$DICOM_ROOT" --mask-root "$MASK_ROOT" \
    --out-root "$AISD_NII" --lesion-values 2 3 4 5

# 3. 转 nnU-Net 数据集（先在 Task402_AIS.py 改 train_dir=$AISD_NII 和 output_folder）
python "$REPO_ROOT/nnUNet/nnunet/dataset_conversion/Task402_AIS.py"

# 4. 计划 + 预处理
nnUNet_plan_and_preprocess -t 402 --verify_dataset_integrity

# 5. 使用论文的数据划分
cp "$REPO_ROOT/data/splits_final.pkl" "$NNUNET_PREPROCESSED/Task402_AIS/splits_final.pkl"

# 6. 训练
(cd "$REPO_ROOT/ClSeg_package/ClSeg/run" && \
 python run_training.py -network_trainer nnUNetTrainerV2_AISD -gpu "$GPU" -task 402 -outpath AISD)

# 7. 推理（验证/测试集，输出预测 nii.gz 与 summary.json）
(cd "$REPO_ROOT/ClSeg_package/ClSeg/run" && \
 python run_training.py -network_trainer nnUNetTrainerV2_AISD -gpu "$GPU" -task 402 -outpath AISD \
     -val --val_folder validation_output)

# 8. 叠加可视化（示例：对某一例；把 <pid> 与 results 路径换成实际值）
# python "$REPO_ROOT/reproduce/overlay_seg.py" \
#     --ct "$AISD_NII/<pid>/CT.nii.gz" \
#     --pred /your/path/results/.../AISD_IN_LeakyReLU/validation_output/<pid>.nii.gz \
#     --gt "$AISD_NII/<pid>/GT_hard.nii.gz" \
#     --out-dir /your/path/overlay/<pid> --only-lesion

echo "流程完成。"
