Core
====

Detector Blueprints
-------------------

.. currentmodule:: nndet.core

.. autosummary::
   :toctree: Detector Blueprints
   :nosignatures:

   retina
   rcnn
   detr

Postprocessing
--------------

.. currentmodule:: nndet.core.post

.. autosummary::
   :toctree: Postprocessing
   :nosignatures:

   box
   mask
   detr


Boxes
-----

Anchor Generators
~~~~~~~~~~~~~~~~~

.. currentmodule:: nndet.core.boxes

.. autosummary::
   :toctree: Anchor Generators
   :nosignatures:

   anchors

Matcher
~~~~~~~

.. currentmodule:: nndet.core.boxes.matcher

.. autosummary::
   :toctree: Matcher
   :nosignatures:

   base
   iou
   atss


.. currentmodule:: nndet.core.boxes

.. autosummary::
   :toctree: Matcher
   :nosignatures:

   assign

Matcher1to1
~~~~~~~~~~~

.. currentmodule:: nndet.core.boxes.matcher1to1

.. autosummary::
   :toctree: Matcher1to1
   :nosignatures:

   base
   hungarian


.. currentmodule:: nndet.core.boxes.criterions

.. autosummary::
   :toctree: Matcher1to1
   :nosignatures:

   base
   box
   cls

Coder
~~~~~

.. currentmodule:: nndet.core.boxes

.. autosummary::
   :toctree: Coder
   :nosignatures:

   coder


NMS
~~~

.. currentmodule:: nndet.core.boxes

.. autosummary::
   :toctree: NMS
   :nosignatures:

    nms
    wbc


Sampler
~~~~~~~

.. currentmodule:: nndet.core.boxes

.. autosummary::
   :toctree: Sampler
   :nosignatures:

   sampler


Ops Torch
~~~~~~~~~

.. currentmodule:: nndet.core

.. autosummary::
   :toctree: Ops Torch
   :nosignatures:

   ops_torch


Ops Numpy
~~~~~~~~~

.. currentmodule:: nndet.core

.. autosummary::
   :toctree: Ops Numpy
   :nosignatures:

   ops_np


RoI
---

RoI Module
~~~~~~~~~~

.. currentmodule:: nndet.core.rois.module

.. autosummary::
   :toctree: RoI Module
   :nosignatures:

   base
   single
   cascade


RoI Pooler
~~~~~~~~~~

.. currentmodule:: nndet.core.rois.pooler

.. autosummary::
   :toctree: RoI Pooler
   :nosignatures:

   base
   roi_align
