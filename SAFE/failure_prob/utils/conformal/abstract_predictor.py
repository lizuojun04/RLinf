from abc import ABC, abstractmethod

import numpy as np


class AbstractPredictor(ABC):
    @abstractmethod
    def get_prediction_band(self, training_data: np.ndarray, calibration_data: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def get_one_sided_prediction_band(self, training_data: np.ndarray, calibration_data: np.ndarray, alpha: float, lower_bound: bool) -> np.ndarray:
        raise NotImplementedError
