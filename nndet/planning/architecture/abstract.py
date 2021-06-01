from abc import ABC, abstractmethod
from typing import TypeVar


class ArchitecturePlanner(ABC):
    def __init__(self, **kwargs):
        """
        Plan architecture and training hyperparameters (batch size and patch size)
        """
        for key, item in kwargs.items():
            setattr(self, key, item)

    @abstractmethod
    def plan(self, *args, **kwargs) -> dict:
        """
        Plan architecture and training parameters

        Args:
            *args: positional arguments determined by Planner
            **kwargs: keyword arguments determined by Planner

        Returns:
            dict: training and architecture information
                `patch_size` (Sequence[int]): patch size
                `batch_size` (int): batch size for training
                `architecture` (dict): dictionary with all parameters needed for the final model
        """
        raise NotImplementedError

    def approximate_vram(self):
        """
        Approximate vram usage of model for planning
        """
        pass
    
    def get_planner_id(self) -> str:
        """
        Create identifier for this planner
        
        Returns:
            str: identifier
        """
        return self.__class__.__name__


ArchitecturePlannerType = TypeVar('ArchitecturePlannerType', bound=ArchitecturePlanner)
