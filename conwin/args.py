import dataclasses
import typing


@dataclasses.dataclass
class TrainingArguments:
    num_epochs: typing.Optional[int] = dataclasses.field(
        default=10, metadata={"help": "Number of training epochs"}
    )
    num_tokens: typing.Optional[int] = dataclasses.field(
        default=50_000_000, metadata={"help": "Number of tokens to train on"}
    )
    window_size: typing.Optional[int] = dataclasses.field(
        default=128, metadata={"help": "Window size"}
    )
    batch_size: typing.Optional[int] = dataclasses.field(
        default=128, metadata={"help": "Training batch size"}
    )
    learning_rate: typing.Optional[float] = dataclasses.field(
        default=1e-5, metadata={"help": "Learning rate"}
    )
    warmup_steps: typing.Optional[int] = dataclasses.field(
        default=3_000, metadata={"help": "Number of warmup steps"}
    )
    scheduler: typing.Optional[str] = dataclasses.field(
        default="cosine", metadata={"help": "Scheduler decay"}
    )
    weight_decay: typing.Optional[float] = dataclasses.field(
        default=0.1, metadata={"help": "Weight decay"}
    )
    random_seed: typing.Optional[int] = dataclasses.field(
        default=42, metadata={"help": "Random seed"}
    )
    eval_steps: typing.Optional[int] = dataclasses.field(
        default=1_000, metadata={"help": "Number of steps between evaluations"}
    )
