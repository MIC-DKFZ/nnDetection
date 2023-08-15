User Guide
==========

- Training Time / Training Speed / Benchmark?

Preparing Data Sets
-------------------

Toy Data Set
~~~~~~~~~~~~

Running `nndet_example` will automatically generate an example data set with 3D squares and sqaures with holes which can be used to test the installation or experiment with prototype code (it is still necessary to run the other nndet commands to process/train/predict the data set).

.. code-block: bash

    # create data to test installation/environment (10 train 10 test)
    nndet_example

    # create full data set for prototyping (1000 train 1000 test)
    nndet_example --full [--num_processes]

The `full` problem is very easy and the final results should be near perfect (even after very short training).
After running the generation script follow the `Planning`, `Training` and `Inference` instructions below to construct the whole nnDetection pipeline.



# TODOs
# - all images need to have the same number of modalities
# - images (all modalities) and corresponding label need to have the same size (number of pixels)
# - modalities need to be registered (bias field correction?)


Using nnDetection
-----------------

# TODOs
# - continue training

Advanced Use Cases
------------------

Detection Zoo
~~~~~~~~~~~~~

+--------------------------+------------------------+---------------------------+-------------------------++------------------------------------------------------------------------------+
| **Models**               | **Inputs**             | **Outputs**               | **Config**              || **Command**                                                                  |
+--------------------------+------------------------+---------------------------+-------------------------++------------------------------------------------------------------------------+
+--------------------------+------------------------+---------------------------+-------------------------++------------------------------------------------------------------------------+
|| Retina U-Net V001       | BB + SS                | BB                        | retinaunet_v001         || train=retinaunet_v001                                                        |
+--------------------------+------------------------+---------------------------+-------------------------++------------------------------------------------------------------------------+
+--------------------------+------------------------+---------------------------+-------------------------++------------------------------------------------------------------------------+
|| RetinaNet V002          | BB                     | BB                        | retinaunet_v002         || train=retinaunet_v002                                                        |
+--------------------------+------------------------+---------------------------+-------------------------++------------------------------------------------------------------------------+
|| Faster RCNN V002        | BB                     | BB                        |                         ||                                                                              |
+--------------------------+------------------------+---------------------------+-------------------------++------------------------------------------------------------------------------+
|| Retina U-Net V002       | BB (+ SS)              | BB                        | retinaunet_v002         || train=retinaunet_v002 module=RetinaNetV002                                   |
+--------------------------+------------------------+---------------------------+-------------------------++------------------------------------------------------------------------------+
|| Box Mask RCNN V002      | BB + BI                | BB                        |                         ||                                                                              |
+--------------------------+------------------------+---------------------------+-------------------------++------------------------------------------------------------------------------+
|| Box Mask U-RCNN V002    | BB + BI (+ SS)         | BB                        |                         ||                                                                              |
+--------------------------+------------------------+---------------------------+-------------------------++------------------------------------------------------------------------------+

Legend: BB = Bounding Boxes, BI = Binary Mask, SS = Semantic Segmentation (dervied from instance segmentation mask)

Trainning Different Versions of RetinaU-Net:
# TODO: focal loss training
