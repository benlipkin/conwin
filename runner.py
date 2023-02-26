import dataclasses
import typing

import datasets
import transformers
from accelerate import Accelerator
from transformers import (
    HfArgumentParser,
    AutoConfig,
    AutoModelForPreTraining,
    AutoTokenizer,
)

from conwin.args import TrainingArguments
from conwin.train import Pipeline

datasets.logging.set_verbosity_error()
transformers.logging.set_verbosity_error()


@dataclasses.dataclass
class PipelineArguments:
    model: typing.Optional[str] = dataclasses.field(
        default="gpt2", metadata={"help": "Model name or path"}
    )
    window_size: typing.Optional[int] = dataclasses.field(
        default=128, metadata={"help": "Window size"}
    )
    dataset: typing.Optional[str] = dataclasses.field(
        default="wikitext", metadata={"help": "Dataset name"}
    )
    subset: typing.Optional[str] = dataclasses.field(
        default="wikitext-103-v1", metadata={"help": "Dataset subset"}
    )
    num_tokens: typing.Optional[int] = dataclasses.field(
        default=50_000_000, metadata={"help": "Number of tokens to train on"}
    )
    output_dir: typing.Optional[str] = dataclasses.field(
        default="output", metadata={"help": "Output directory"}
    )


def main():
    args = HfArgumentParser([PipelineArguments, TrainingArguments]).parse_args()
    accelerator = Accelerator()
    dataset = datasets.load_dataset(args.dataset, args.subset)
    config = AutoConfig.from_pretrained(args.model, n_positions=args.window_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForPreTraining.from_config(config)
    Pipeline(accelerator, dataset, tokenizer, model, args).run()


if __name__ == "__main__":
    main()
