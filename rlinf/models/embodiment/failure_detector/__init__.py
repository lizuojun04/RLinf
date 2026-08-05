# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SAFE failure-detection primitives for RLinf rollout scoring.

Re-exports the functional-CP band computation and the online SAFE detector so the
rollout/env workers can do:

    detector = SafeFailureDetector.from_ckpt(ckpt, cfg_infos)
    cp_band, info = compute_safe_band(data_path, detector, alpha)   # in-memory
    score = detector.forward_step(raw_suffix_out)          # (B,)
    flag = score >= cp_band[min(step, T-1)]                 # (B,)

The functional-CP band is always recomputed from the offline calibration rollouts
(``data_path``'s ``**/{success,fail}/*.pkl`` layout, the RLinf-native dump format)
at evaluation time, exactly as the SAFE evaluation does. No static band files are
persisted.
"""

from rlinf.models.embodiment.failure_detector.functional_cp import (
    flag_from_band,
    get_one_sided_prediction_band,
)
from rlinf.models.embodiment.failure_detector.safe_calibration import (
    compute_band_from_calibration,
    compute_safe_band,
    load_safe_rollouts,
)
from rlinf.models.embodiment.failure_detector.safe_detector import (
    SafeFailureDetector,
    preprocess_suffix_out,
)

__all__ = [
    "SafeFailureDetector",
    "preprocess_suffix_out",
    "flag_from_band",
    "get_one_sided_prediction_band",
    "compute_safe_band",
    "compute_band_from_calibration",
    "load_safe_rollouts",
]
