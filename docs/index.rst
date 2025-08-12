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

- Probabilistic segmentation: compute likelihood of modeled cells
- Optimized for large-scale 2D and 3D data
- Roadmap: 4D support in development
- Includes low-level and high-level API

--------------------------------
🚀 How to Use

Use the `SegmentationDataset` class for a structured workflow.  
For custom pipelines, call the low-level segmentation and registration functions directly.

--------------------------------

.. toctree::
   :maxdepth: 2
   :caption: 📚 Documentation:

   api
   tutorial
