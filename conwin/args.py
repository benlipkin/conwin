import dataclasses
import typing


@dataclasses.dataclass
class TrainingArguments:
    num_epochs: typing.Optional[int] = dataclasses.field(
        default=1, metadata={"help": "Number of training epochs"}
    )
    batch_size: typing.Optional[int] = dataclasses.field(
        default=128, metadata={"help": "Training batch size"}
    )
    learning_rate: typing.Optional[float] = dataclasses.field(
        default=5e-5, metadata={"help": "Learning rate"}
    )
    warmup_steps: typing.Optional[int] = dataclasses.field(
        default=1_000, metadata={"help": "Number of warmup steps"}
    )
    weight_decay: typing.Optional[float] = dataclasses.field(
        default=0.1, metadata={"help": "Weight decay"}
    )
    random_seed: typing.Optional[int] = dataclasses.field(
        default=42, metadata={"help": "Random seed"}
    )
    eval_steps: typing.Optional[int] = dataclasses.field(
        default=500, metadata={"help": "Number of steps between evaluations"}
    )
