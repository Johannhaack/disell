.. disell documentation master file, created by
   sphinx-quickstart on Thu Jul 10 12:46:27 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to disell
==================

⚠️ **Note**: Currently supported only on **Linux**.

--------------------------------

**Disell** is a high-performance segmentation framework for Dark Field X-ray Microscopy (DFXM).  
It leverages parallel CPU processing to efficiently segment dislocation cells, even when boundaries are weak or noisy.

--------------------------------
🔍 Key Features

- Methods to identify Dislocation Cells in 2D and 3D 
- Methods to grow these structures into full segmentation mask of the image/volume 

--------------------------------
🚀 How to Use

There is a two level approach first using the methods in cell_idetification.py initial regions of dislocation cells are idetified for this paramter tuning is needed.
In the second level using methods in region_growing.py the initial cells are used to provide a full segmentation mask.

--------------------------------

.. toctree::
   :maxdepth: 2
   :caption: 📚 Documentation:

   api
   tutorial
