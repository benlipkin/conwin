from brainscore_language.benchmarks.pereira2018.benchmark import (
    _Pereira2018ExperimentLinear,
    BIBTEX,
)
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_language import benchmark_registry

import metric

import xarray as xr
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.utils.ceiling import ceiling_normalize
from brainscore_core.metrics import Score


class _Pereira2018ExpPsgSpRgCVLinear(_Pereira2018ExperimentLinear):
    """
    Evaluate model ability to predict neural activity in the human language system in response to natural sentences,
    recorded by Pereira et al. 2018.
    Alignment of neural activity between model and human subjects is evaluated via cross-validated linear predictivity.

    This benchmark builds off the Pereira2018 benchmark introduced
    in Schrimpf et al. 2021 (https://www.pnas.org/doi/10.1073/pnas.2105646118), but:

    * computes neural alignment to each of the two experiments ({243,384}sentences) separately, as well as ceilings
    * requires the model to have committed to neural readouts (e.g. layer 41 corresponds to the language system),
        rather than testing every layer separately

    Each of these benchmarks evaluates one of the two experiments, the overall Pereira2018-linear score is the mean of
    the two ceiling-normalized scores.
    """

    def __init__(self, experiment: str, ceiling_s3_kwargs: dict):
        self.data = self._load_data(experiment)
        self.metric = metric.rgcv_linear_pearsonr(
            crossvalidation_kwargs=dict(
                split_coord="passage_label",
                train_size=0.9,
                test_size=None,
                unique_split_values=True,
            )
        )
        ceiling = self._load_ceiling(
            identifier=f"Pereira2018.{experiment}-linear", **ceiling_s3_kwargs
        )
        identifier = f"Pereira2018.{experiment}-psgsp_rgcv_linear"
        BenchmarkBase.__init__(
            self,
            identifier=identifier,
            version=1,
            parent="Pereira2018-linear",
            ceiling=ceiling,
            bibtex=BIBTEX,
        )

    def __call__(self, candidate: ArtificialSubject) -> Score:
        candidate.start_neural_recording(
            recording_target=ArtificialSubject.RecordingTarget.language_system,
            recording_type=ArtificialSubject.RecordingType.fMRI,
        )
        stimuli = self.data["stimulus"]
        passages = self.data["passage_label"].values
        predictions = []
        for passage in sorted(
            set(passages)
        ):  # go over individual passages, sorting to keep consistency across runs
            passage_indexer = [
                stimulus_passage == passage for stimulus_passage in passages
            ]
            passage_stimuli = stimuli[passage_indexer]
            passage_predictions = candidate.digest_text(passage_stimuli.values)[
                "neural"
            ]
            passage_predictions["stimulus_id"] = (
                "presentation",
                passage_stimuli["stimulus_id"].values,
            )
            passage_predictions["passage_label"] = (
                "presentation",
                passage_stimuli["passage_label"].values,
            )
            predictions.append(passage_predictions)
        predictions = xr.concat(predictions, dim="presentation")
        raw_score = self.metric(predictions, self.data)
        score = ceiling_normalize(raw_score, self.ceiling)
        return score


def Pereira2018_243sentences_psgsp_rgcv():
    return _Pereira2018ExpPsgSpRgCVLinear(
        experiment="243sentences",
        ceiling_s3_kwargs=dict(
            version_id="CHl_9aFHIWVnPW_njePfy28yzggKuUPw",
            sha1="5e23de899883828f9c886aec304bc5aa0f58f66c",
            raw_kwargs=dict(
                version_id="uZye03ENmn.vKB5mARUGhcIY_DjShtPD",
                sha1="525a6ac8c14ad826c63fdd71faeefb8ba542d5ac",
                raw_kwargs=dict(
                    version_id="XVTo58Po5YrNjTuDIWrmfHI0nbN2MVZa",
                    sha1="34ba453dc7e8a19aed18cc9bca160e97b4a80be5",
                ),
            ),
        ),
    )


def Pereira2018_384sentences_psgsp_rgcv():
    return _Pereira2018ExpPsgSpRgCVLinear(
        experiment="384sentences",
        ceiling_s3_kwargs=dict(
            version_id="sjlnXr5wXUoGv6exoWu06C4kYI0KpZLk",
            sha1="fc895adc52fd79cea3040961d65d8f736a9d3e29",
            raw_kwargs=dict(
                version_id="Hi74r9UKfpK0h0Bjf5DL.JgflGoaknrA",
                sha1="ce2044a7713426870a44131a99bfc63d8843dae0",
                raw_kwargs=dict(
                    version_id="m4dq_ouKWZkYtdyNPMSP0p6rqb7wcYpi",
                    sha1="fe9fb24b34fd5602e18e34006ac5ccc7d4c825b8",
                ),
            ),
        ),
    )


benchmark_registry[
    "Pereira2018.243sentences-psgsp_rgcvlinear"
] = Pereira2018_243sentences_psgsp_rgcv
benchmark_registry[
    "Pereira2018.384sentences-psgsp_rgcvlinear"
] = Pereira2018_384sentences_psgsp_rgcv