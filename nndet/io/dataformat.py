# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

# Acknowledgements: The following changes in this codebase have been implemented following the updates and improvements made in the nnU-Net repository.
# nnU-Net repository: https://github.com/MIC-DKFZ/nnUNet

from copy import deepcopy
import math
import numpy as np
import blosc2
from typing import Union, Tuple
from abc import ABC, abstractmethod

class PreprocessedDataset(ABC):
    
    def __init__(self, file_extension: str):
        self.file_extension = file_extension

    @abstractmethod
    def load_data(self, path: str, is_path_truncated: bool = False):
        pass

    @abstractmethod
    def load_seg(self, path: str, is_path_truncated: bool = False):
        pass

    @abstractmethod
    def save(
        self,
        truncated_path: str,
        data: np.ndarray,
        seg: np.ndarray,
        **kwargs
    ):
        pass
    
    def get_file_extension(self):
        return self.file_extension
    

class PreprocessedDatasetNumpy(PreprocessedDataset):
    def __init__(self) -> None:
        super().__init__(file_extension="npz")
    
    def load_data(self, path: str, is_path_truncated: bool = False):
        if is_path_truncated:
            data = np.load(f"{path}.npz", mmap_mode='r')
        else:
            data = np.load(path, mmap_mode='r')
            
        if type(data) == np.memmap:
            return data
        else:
            return data['data']
    
    def load_seg(self, path: str, is_path_truncated: bool = False):
        if is_path_truncated:
            data = np.load(f"{path}.npz", mmap_mode='r')
        else:
            data = np.load(path, mmap_mode='r')
        
        if type(data) == np.memmap:
            return data
        else:
            return data['seg']
        
        
    def save(
        self,
        truncated_path: str,
        data: np.ndarray,
        seg: np.ndarray,
        **kwargs
    ):
        np.savez_compressed(truncated_path + ".npz", data=data, seg=seg)
    
    
class PreprocessedDatasetBlosc2(PreprocessedDataset):
    
    def __init__(self):
        super().__init__(file_extension="b2nd")
        self.block_size = None
        self.chunk_size = None
        self.seg_block_size = None
        self.seg_chunk_size = None
        
        self.data_size = None
        self.seg_size = None
        self.patch_size = None
        blosc2.set_nthreads(1)
    
    def load_data(self, path: str, is_path_truncated: bool = False):
        if is_path_truncated:
            return blosc2.open(urlpath=f"{path}.b2nd", mode='r')
        else:
            return blosc2.open(urlpath=path, mode='r')
    
    def load_seg(self, path: str, is_path_truncated: bool = False):
        if is_path_truncated:
            return blosc2.open(urlpath=f"{path}_seg.b2nd", mode='r')
        else:
            return blosc2.open(urlpath=path, mode='r')

    def save(
        self,
        truncated_path: str,
        data: np.ndarray,
        seg: np.ndarray,
        patch_size: Union[Tuple[int, int], Tuple[int, int, int]]
    ):
        if (self.data_size != data.shape) or (self.seg_size != seg.shape) or (self.patch_size != patch_size):
            self.data_size = data.shape
            self.seg_size = seg.shape
            self.patch_size = patch_size
            self.block_size, self.chunk_size = self.comp_blosc2_params(self.data_size, self.patch_size, data.itemsize)
            self.seg_block_size, self.seg_chunk_size = self.comp_blosc2_params(self.seg_size, self.patch_size, seg.itemsize)

        cparams = {
            'codec': blosc2.Codec.ZSTD,
            # 'filters': [blosc2.Filter.SHUFFLE],
            # 'splitmode': blosc2.SplitMode.ALWAYS_SPLIT,
            'clevel': 8,
        }
        blosc2.asarray(np.ascontiguousarray(data), urlpath=truncated_path + '.b2nd', 
                       chunks=self.chunk_size, blocks=self.block_size, 
                       cparams=cparams, mode='w')
        blosc2.asarray(np.ascontiguousarray(seg), urlpath=truncated_path + '_seg.b2nd', 
                       chunks=self.seg_chunk_size, blocks=self.seg_block_size, 
                       cparams=cparams, mode='w')
    
    @staticmethod
    def comp_blosc2_params(
        image_size: Tuple[int, int, int, int],
        patch_size: Union[Tuple[int, int], Tuple[int, int, int]],
        bytes_per_pixel: int = 4,  # 4 byte are float32
        l1_cache_size_per_core_in_bytes: int = 32768,  # 1 Kibibyte (KiB) = 2^10 Byte;  32 KiB = 32768 Byte
        l3_cache_size_per_core_in_bytes: int = 1441792,
        # 1 Mibibyte (MiB) = 2^20 Byte = 1.048.576 Byte; 1.375MiB = 1441792 Byte
        safety_factor: float = 0.8  # we dont will the caches to the brim. 0.8 means we target 80% of the caches
    ):
        """
        Computes a recommended block and chunk size for saving arrays with blosc v2.

        Bloscv2 NDIM doku: "Remember that having a second partition means that we have better flexibility to fit the
        different partitions at the different CPU cache levels; typically the first partition (aka chunks) should
        be made to fit in L3 cache, whereas the second partition (aka blocks) should rather fit in L2/L1 caches
        (depending on whether compression ratio or speed is desired)."
        (https://www.blosc.org/posts/blosc2-ndim-intro/)
        -> We are not 100% sure how to optimize for that. For now we try to fit the uncompressed block in L1. This
        might spill over into L2, which is fine in our books.

        Note: this is optimized for nnU-Net dataloading where each read operation is done by one core. We cannot use threading

        Cache default values computed based on old Intel 4110 CPU with 32K L1, 128K L2 and 1408K L3 cache per core.
        We cannot optimize further for more modern CPUs with more cache as the data will need be be read by the
        old ones as well.

        Args:
            patch_size: Image size, must be 4D (c, x, y, z). For 2D images, make x=1
            patch_size: Patch size, spatial dimensions only. So (x, y) or (x, y, z)
            bytes_per_pixel: Number of bytes per element. Example: float32 -> 4 bytes
            l1_cache_size_per_core_in_bytes: The size of the L1 cache per core in Bytes.
            l3_cache_size_per_core_in_bytes: The size of the L3 cache exclusively accessible by each core. Usually the global size of the L3 cache divided by the number of cores.

        Returns:
            The recommended block and the chunk size.
        """
        # Fabians code is ugly, but eh

        num_channels = image_size[0]
        if len(patch_size) == 2:
            patch_size = [1, *patch_size]
        patch_size = np.array(patch_size)
        block_size = np.array((num_channels, *[2 ** (max(0, math.floor(math.log2(i / 2)))) for i in patch_size]))

        # shrink the block size until it fits in L1
        estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel
        while estimated_nbytes_block > (l1_cache_size_per_core_in_bytes * safety_factor):
            # pick largest deviation from patch_size that is not 1
            axis_order = np.argsort(block_size[1:] / patch_size)[::-1]
            idx = 0
            picked_axis = axis_order[idx]
            while block_size[picked_axis + 1] == 1 or block_size[picked_axis + 1] == image_size[picked_axis + 1]:
                idx += 1
                picked_axis = axis_order[idx]
            # now reduce that axis to the next lowest power of 2
            block_size[picked_axis + 1] = 2 ** (max(0, math.floor(math.log2(block_size[picked_axis + 1] - 1))))
            block_size[picked_axis + 1] = min(block_size[picked_axis + 1], image_size[picked_axis + 1])
            estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel
            if all([i == j for i, j in zip(block_size, image_size)]):
                break

        # note: there is no use extending the chunk size to 3d when we have a 2d patch size! This would unnecessarily
        # load data into L3
        # now tile the blocks into chunks until we hit image_size or the l3 cache per core limit
        chunk_size = deepcopy(block_size)
        estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
        while estimated_nbytes_chunk < (l3_cache_size_per_core_in_bytes * safety_factor):
            # find axis that deviates from block_size the most
            axis_order = np.argsort(chunk_size[1:] / block_size[1:])
            idx = 0
            picked_axis = axis_order[idx]
            while chunk_size[picked_axis + 1] == image_size[picked_axis + 1] or patch_size[picked_axis] == 1:
                idx += 1
                picked_axis = axis_order[idx]
            chunk_size[picked_axis + 1] += block_size[picked_axis + 1]
            chunk_size[picked_axis + 1] = min(chunk_size[picked_axis + 1], image_size[picked_axis + 1])
            estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
            if patch_size[0] == 1:
                if all([i == j for i, j in zip(chunk_size[2:], image_size[2:])]):
                    break
            if all([i == j for i, j in zip(chunk_size, image_size)]):
                break
        # print(image_size, chunk_size, block_size)
        return tuple(block_size), tuple(chunk_size)

    
data_format_to_class_mapping = {
    "npz" : PreprocessedDatasetNumpy(),
    "b2nd": PreprocessedDatasetBlosc2()
}