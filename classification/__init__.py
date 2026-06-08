"""AISD 2D slice-level stroke lesion classification.

Three classes:
    0 - no stroke  (mask has no lesion pixels on the slice)
    1 - small lesion
    2 - large lesion

The pipeline is self-contained (PyTorch + numpy + Pillow + torchvision) and does
NOT depend on the nnU-Net / ClSeg segmentation training framework. Labels are
derived automatically from the segmentation masks. See README.md for usage.
"""
