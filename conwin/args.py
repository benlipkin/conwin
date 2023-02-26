import dataclasses
import typing


@dataclasses.dataclass
class TrainingArguments:
    learning_rate: typing.Optional[float] = dataclasses.field(
        default=5e-5, metadata={"help": "Learning rate"}
    )
    warmup_steps: typing.Optional[int] = dataclasses.field(
        default=1_000, metadata={"help": "Number of warmup steps"}
    )
    batch_size: typing.Optional[int] = dataclasses.field(
        default=128, metadata={"help": "Training batch size"}
    )
    num_epochs: typing.Optional[int] = dataclasses.field(
        default=1, metadata={"help": "Number of training epochs"}
    )
    random_seed: typing.Optional[int] = dataclasses.field(
        default=42, metadata={"help": "Random seed"}
    )
    eval_steps: typing.Optional[int] = dataclasses.field(
        default=1_000, metadata={"help": "Number of steps between evaluations"}
    )
