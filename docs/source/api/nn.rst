Neural Networks
===============

Backbones
---------

.. currentmodule:: nndet.nn.backbone

.. autosummary::
   :toctree: Backbones
   :nosignatures:

   abstract


.. currentmodule:: nndet.nn.backbone.blueprints

.. autosummary::
   :toctree: Backbones
   :nosignatures:

   level
   conv


.. currentmodule:: nndet.nn.backbone.statics

.. autosummary::
   :toctree: Backbones
   :nosignatures:

   resnet
   swin


Necks
-----

.. currentmodule:: nndet.nn.neck

.. autosummary::
   :toctree: Necks
   :nosignatures:

   abstract
   fpn


Classifier Heads
----------------

.. currentmodule:: nndet.nn.heads.classifier

.. autosummary::
   :toctree: Classifier Heads
   :nosignatures:

   dense
   roi


Regression Heads
----------------

.. currentmodule:: nndet.nn.heads.regressor

.. autosummary::
   :toctree: Regression Heads
   :nosignatures:

   dense
   roi


Comb Heads
----------

.. currentmodule:: nndet.nn.heads.comb

.. autosummary::
   :toctree: Comb Heads
   :nosignatures:

   base
   roi
   anchor_all
   anchor_sampled


Mask Heads
----------

.. currentmodule:: nndet.nn.heads.masker

.. autosummary::
   :toctree: Mask Heads
   :nosignatures:

   base


Segmentation Heads
------------------

.. currentmodule:: nndet.nn.heads

.. autosummary::
   :toctree: Segmentation Heads
   :nosignatures:

   segmenter

Layers
------

.. currentmodule:: nndet.nn.layers

.. autosummary::
   :toctree: Layers
   :nosignatures:

   initializer
   wrapper


.. currentmodule:: nndet.nn.layers.conv

.. autosummary::
   :toctree: Layers
   :nosignatures:

   base
   batch
   group
   instance


Ops
---

.. currentmodule:: nndet.nn.ops

.. autosummary::
   :toctree: Layers
   :nosignatures:

   activation
   interpolation
   norm
   scale
