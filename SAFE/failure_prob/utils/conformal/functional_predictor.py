from enum import Enum

import numpy as np

from .abstract_predictor import AbstractPredictor


class RegressionType(Enum):
    Mean = 1


class ModulationType(Enum):
    Const = 1
    Stdev = 2
    Tfunc = 3


def regress(training_data: np.ndarray, regression_type: RegressionType) -> np.ndarray:
    if regression_type == RegressionType.Mean:
        return np.mean(training_data, axis=0, keepdims=True)
    raise NotImplementedError


class FunctionalPredictor(AbstractPredictor):
    def __init__(self, modulation_type: ModulationType, regression_type: RegressionType):
        self.modulation_type = modulation_type
        self.regression_type = regression_type

    def get_prediction_band(self, training_data: np.ndarray, calibration_data: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        length = training_data.shape[-1]
        assert length == calibration_data.shape[-1]
        assert 0.0 < alpha < 1.0

        prediction_trajectory = regress(training_data, self.regression_type)
        modulation_trajectory = self._get_modulation_trajectory(training_data, prediction_trajectory, alpha)

        calibration_scores = [
            np.max(np.abs(calibration_trajectory - prediction_trajectory) / modulation_trajectory)
            for calibration_trajectory in calibration_data
        ]
        calibration_size = len(calibration_scores)
        band_width = np.sort(calibration_scores)[int(np.ceil((calibration_size + 1) * (1 - alpha))) - 1]
        upper_trajectory = prediction_trajectory + band_width * modulation_trajectory
        lower_trajectory = prediction_trajectory - band_width * modulation_trajectory
        assert upper_trajectory.shape == lower_trajectory.shape == (1, length)
        return (upper_trajectory, lower_trajectory)

    def get_one_sided_prediction_band(self, training_data: np.ndarray, calibration_data: np.ndarray, alpha: float, lower_bound: bool) -> np.ndarray:
        length = training_data.shape[-1]
        assert length == calibration_data.shape[-1]
        assert 0.0 < alpha < 1.0

        prediction_trajectory = regress(training_data, self.regression_type)
        modulation_trajectory = self._get_modulation_trajectory(training_data, prediction_trajectory, alpha)

        if lower_bound:
            calibration_scores = [
                np.max((prediction_trajectory - calibration_trajectory) / modulation_trajectory)
                for calibration_trajectory in calibration_data
            ]
        else:
            calibration_scores = [
                np.max((calibration_trajectory - prediction_trajectory) / modulation_trajectory)
                for calibration_trajectory in calibration_data
            ]
        band_width = np.quantile(calibration_scores, 1 - alpha)

        if lower_bound:
            bounding_trajectory = prediction_trajectory - band_width * modulation_trajectory
        else:
            bounding_trajectory = prediction_trajectory + band_width * modulation_trajectory
        return bounding_trajectory

    def _get_modulation_trajectory(self, training_data: np.ndarray, prediction_trajectory: np.ndarray, alpha: float) -> np.ndarray:
        eps = 1e-8
        length = training_data.shape[-1]
        if self.modulation_type == ModulationType.Const:
            modulation_trajectory = np.ones((1, length)) / length
        elif self.modulation_type == ModulationType.Stdev:
            modulation_trajectory = np.std(training_data, axis=0, ddof=1, keepdims=True) + eps
        elif self.modulation_type == ModulationType.Tfunc:
            train_size = training_data.shape[0]
            if int(np.ceil((train_size + 1) * (1 - alpha))) > train_size:
                modulation_trajectory = np.max(np.abs(training_data - prediction_trajectory), axis=0, keepdims=True) + eps
            else:
                gamma = np.sort(np.max(np.abs(training_data - prediction_trajectory), axis=1))[
                    int(np.ceil((train_size + 1) * (1 - alpha))) - 1
                ]
                modulation_trajectory = np.max(
                    np.abs(training_data - prediction_trajectory)[
                        np.max(np.abs(training_data - prediction_trajectory), axis=1) <= gamma
                    ],
                    axis=0, keepdims=True,
                ) + eps
        else:
            raise NotImplementedError
        return modulation_trajectory
