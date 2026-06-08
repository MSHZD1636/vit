"""AISD 3D lesion-size binary classification (small_lesion / large_lesion).

This module performs *binary lesion-size* classification on 3D data and reuses
the paper's CNN-Transformer hybrid encoder (``Coformer3D`` from the Cl-SegNet
segmentation network) as the backbone, with a global-pool + linear head.

    0 - small_lesion  (lesion-bearing 3D sub-volume, lesion volume <= threshold)
    1 - large_lesion  (lesion-bearing 3D sub-volume, lesion volume >  threshold)

Only lesion-bearing 3D sub-volumes are used; sub-volumes without any lesion are
dropped (there is no "no stroke" class here). The default sampling mode is
"slab": each sample is a stack of K consecutive axial slices. A "volume" mode
(one sample per patient, sized by the whole-case lesion volume) is also provided.

The 2D slice-level 3-class pipeline in ``classification/`` is left untouched.
"""
