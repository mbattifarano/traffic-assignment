from abc import ABC, abstractmethod


class StepSize(ABC):

    @abstractmethod
    def step(self, k: int) -> float:
        pass


class MonotoneDecreasingStepSize(StepSize):

    def step(self, k: int) -> float:
        return 2 / (k + 2)
