from abc import ABC, abstractmethod
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray


class Action(IntEnum):
    NO_OP = 0
    LEFT = 1
    RIGHT = 2
    DOWN = 3
    UP = 4

class Policy(ABC):
    @abstractmethod
    def sample(self, obs: NDArray[np.float32]) -> int:
        ...
