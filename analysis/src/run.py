import collections
import multiprocessing
import pathlib

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend

import pereira

from brainscore_language import benchmark_registry, load_benchmark
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from brainscore_language import ArtificialSubject

MODELS = [
    "benlipkin/gpt2_8_wikitext_100M_20_27a3016f17f9dd51",
    "benlipkin/gpt2_16_wikitext_100M_20_1c15056cf51bff47",
    "benlipkin/gpt2_32_wikitext_100M_20_4271d55d34c8c387",
    "benlipkin/gpt2_64_wikitext_100M_20_5cd4da41b7fe7e3d",
    "benlipkin/gpt2_128_wikitext_100M_20_6adb2593f59e6343",
    "benlipkin/gpt2_256_wikitext_100M_20_26e50955232e9b5c",
    "benlipkin/gpt2_512_wikitext_100M_20_d4f8870be67f0770",
    "benlipkin/gpt2_1024_wikitext_100M_20_e12e6d4615e6a1e5",
]

BENCHMARKS = [
    # "Pereira2018.243sentences-psgsp_rgcvlinear",
    # "Pereira2018.384sentences-psgsp_rgcvlinear",
    "Futrell2018-pearsonr",
]

LAYERS = [f"transformer.h.{block}" for block in range(12)]

CONTEXT = [0, 8, 16, 32, 64, 128, 256, 512]


def main(context):
    fname = f"../scores/results_{context}_futrell.csv"
    if pathlib.Path(fname).exists():
        return
    table = collections.defaultdict(list)
    for benchmark_id in tqdm(BENCHMARKS, desc="benchmark"):
        try:
            benchmark = load_benchmark(benchmark_id)
        except:
            benchmark = benchmark_registry[benchmark_id]()
        for model_id in tqdm(MODELS, desc="model"):
            conwin = int(model_id.split("_")[1])
            if conwin < context and context != 0:
                continue
            for layer_id in tqdm(LAYERS, desc="layer"):
                layer_model = HuggingfaceSubject(
                    model_id=model_id,
                    region_layer_mapping={
                        ArtificialSubject.RecordingTarget.language_system: layer_id
                    },
                )
                if context != 0:
                    layer_model.tokenizer.model_max_length = context
                score_xr = benchmark(layer_model)
                scores = list(score_xr.raw.raw.mean("split").values)
                table["benchmark"].append(benchmark_id)
                table["model"].append(model_id)
                table["layer"].append(layer_id)
                table["scores"].append(scores)
    table = pd.DataFrame(table)
    table.to_csv(fname, index=False)


if __name__ == "__main__":
    with parallel_backend("loky", n_jobs=multiprocessing.cpu_count()):
        Parallel()(delayed(main)(context) for context in CONTEXT)
