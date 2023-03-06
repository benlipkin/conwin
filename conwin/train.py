import argparse
import itertools
import json
import shutil
import time

import accelerate
import datasets
import torch
import transformers
from huggingface_hub import Repository, create_repo, get_full_repo_name
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from transformers import get_scheduler
import wandb

from conwin.abstract import Object


class Pipeline(Object):
    def __init__(
        self,
        accelerator: accelerate.Accelerator,
        dataset: datasets.DatasetDict,
        tokenizer: transformers.PreTrainedTokenizerFast,
        model: transformers.AutoModelForPreTraining,
        args: argparse.Namespace,
    ) -> None:
        super().__init__()
        self._accelerator = accelerator
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._model = model
        self._output_dir = self._base / args.output_dir / args.id
        self._args = args
        self._repo: Repository
        self._train_dataloader: DataLoader
        self._eval_dataloader: DataLoader
        self._optimizer: AdamW
        self._scheduler: transformers.trainer_utils.SchedulerType
        self._total_steps: int
        self._best_loss: float = float("inf")

    def _prepare_repo(self):
        self.info("Preparing Repository...")

        def human_format(num: int | float) -> str:
            num = float(f"{num:.3g}")
            mag = 0
            while abs(num) >= 1000:
                mag += 1
                num /= 1000.0
            return f"{str(num).rstrip('.0')}{['', 'K', 'M', 'B', 'T'][mag]}"

        model_id = "_".join(
            str(x)
            for x in [
                self._args.model,
                self._args.window_size,
                self._args.dataset,
                human_format(self._args.num_tokens),
                self._args.num_epochs,
                self._args.id,
            ]
        )
        repo_id = get_full_repo_name(model_id)
        create_repo(repo_id, repo_type="model", exist_ok=True)
        self._repo = Repository(self._output_dir, clone_from=repo_id)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save_pretrained(self._output_dir)
        with open(self._output_dir / "args.json", "w", encoding="utf-8") as f:
            json.dump(vars(self._args), f, indent=4)
        self._repo.push_to_hub(commit_message="args and setup.", blocking=False)

    def _prepare_data(self):
        self.info("Preparing Dataset...")

        def tokenize(sample: datasets.DatasetDict):
            outputs = self._tokenizer(
                sample["text"],
                truncation=True,
                max_length=self._args.window_size,
                return_overflowing_tokens=True,
                return_length=True,
            )
            input_batch = []
            for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
                if length == self._args.window_size:
                    input_batch.append(input_ids)
            return {"input_ids": input_batch}

        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._dataset = self._dataset.map(tokenize, batched=True, remove_columns="text")
        self._dataset["train"] = (
            self._dataset["train"]
            .shuffle(seed=self._args.random_seed)
            .select(range(self._args.num_tokens // self._args.window_size + 1))
        )
        self._dataset.set_format("torch")
        self._train_dataloader = DataLoader(
            self._dataset["train"], batch_size=self._args.batch_size
        )
        self._eval_dataloader = DataLoader(
            self._dataset["validation"], batch_size=self._args.batch_size
        )
        self._total_steps = self._args.num_epochs * len(self._train_dataloader)

    def _prepare_optimizer(self):
        self.info("Preparing Optimizer...")
        params_wd, params_nd = [], []
        for n, p in self._model.named_parameters():
            if any(nd in n for nd in ["bias", "LayerNorm.weight"]):
                params_nd.append(p)
            else:
                params_wd.append(p)
        self._optimizer = AdamW(
            [
                {"params": params_wd, "weight_decay": self._args.weight_decay},
                {"params": params_nd, "weight_decay": 0.0},
            ],
            lr=self._args.learning_rate,
        )
        self._scheduler = get_scheduler(
            name=self._args.scheduler,
            optimizer=self._optimizer,
            num_warmup_steps=self._args.warmup_steps,
            num_total_steps=self._total_steps,
        )

    def _accelerate(self):
        self.info("Preparing Accelerator...")
        (
            self._model,
            self._optimizer,
            self._scheduler,
            self._train_dataloader,
            self._eval_dataloader,
        ) = self._accelerator.prepare(
            self._model,
            self._optimizer,
            self._scheduler,
            self._train_dataloader,
            self._eval_dataloader,
        )

    def _evaluate(self):
        losses = []
        for _, batch in enumerate(self._eval_dataloader):
            with torch.no_grad():
                outputs = self._model(batch["input_ids"], labels=batch["input_ids"])
            losses.append(self._accelerator.gather(outputs.loss))
        loss = torch.mean(torch.stack(losses))
        try:
            perplexity = torch.exp(loss)
        except OverflowError:
            perplexity = float("inf")
        return loss.item(), perplexity.item()

    def _log_callback(self, epoch: int, step: int, loss: float, start: float):
        if self._accelerator.is_main_process:
            self.info(
                f"Time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}, "
                + f"Epoch: {epoch}, Step: {step}, Step Loss: {loss:.2f}"
            )

    def _eval_callback(
        self, epoch: int, step: int, step_loss: float, save: bool = False
    ):
        self._model.eval()
        loss, ppl = self._evaluate()
        if self._accelerator.is_main_process:
            self.info(f"Eval Loss: {loss:.2f}, Perplexity: {ppl:.2f}")
            wandb.log(
                {
                    "epoch": epoch,
                    "step": step,
                    "step_loss": step_loss,
                    "eval_loss": loss,
                    "ppl": ppl,
                }
            )
            if loss < self._best_loss:
                self._best_loss = loss
                self._accelerator.wait_for_everyone()
                self._accelerator.unwrap_model(self._model).save_pretrained(
                    self._output_dir
                )
            if save:
                shutil.move(self._base / "train.log", self._output_dir)
                self._repo.push_to_hub(commit_message="done training.", blocking=False)
        self._model.train()

    def _take_optim_step(self, loss: torch.Tensor):
        self._accelerator.backward(loss)
        self._accelerator.clip_grad_norm_(self._model.parameters(), 1.0)
        self._optimizer.step()
        self._scheduler.step()
        self._optimizer.zero_grad()

    def _train(self):
        self.info("Initializing Training Loop...")
        self.info(f"Total Steps: {self._total_steps}")
        self._model.train()
        start = time.time()
        for step, (epoch, batch) in enumerate(
            itertools.product(
                range(1, self._args.num_epochs + 1), self._train_dataloader
            ),
            start=1,
        ):
            outputs = self._model(batch["input_ids"], labels=batch["input_ids"])
            loss = outputs.loss
            if step % (self._args.eval_steps // 10) == 0:
                self._log_callback(epoch, step, loss.item(), start)
            self._take_optim_step(loss)
            if (step == 1) | (step % self._args.eval_steps == 0):
                self._eval_callback(epoch, step, loss.item())
        self._eval_callback(
            self._args.num_epochs, self._total_steps, loss.item(), save=True
        )

    def run(self):
        wandb.init(project="conwin", config=vars(self._args), name=self._args.id)
        self._prepare_repo()
        self._prepare_data()
        self._prepare_optimizer()
        self._accelerate()
        self._train()
        wandb.finish()
