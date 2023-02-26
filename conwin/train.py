import argparse

import accelerate
import datasets
import torch
import transformers
from huggingface_hub import Repository, get_full_repo_name
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from transformers import get_scheduler

from conwin.abstract import Object

# integrate with wandb logging
# setup to push checkpoints to hub


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
        self._output_dir = self._base / args.output_dir
        self._args = args

    def _prepare_repo(self):
        model_id = "_".join(
            str(x)
            for x in [
                self._args.model,
                self._args.window_size,
                self._args.dataset,
                self._args.num_tokens,
            ]
        )
        repo_id = get_full_repo_name(model_id)
        # self._repo = Repository(self._output_dir, clone_from=repo_id)

    def _prepare_data(self):
        def _tokenize(sample):
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

        self.info("Tokenizing dataset...")
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._dataset = self._dataset.map(
            _tokenize, batched=True, remove_columns="text"
        )
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

    def _prepare_optimizer(self):
        self._optimizer = AdamW(self._model.parameters(), lr=self._args.learning_rate)
        self._scheduler = get_scheduler(
            name="linear",
            optimizer=self._optimizer,
            num_warmup_steps=self._args.warmup_steps,
            num_training_steps=self._args.num_epochs * len(self._train_dataloader),
        )

    def _accelerate(self):
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
        self._model.eval()
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

    def _train(self):
        self.info("Initializing Training Loop...")
        self.info(f"{self._args.num_epochs} epochs")
        self.info(f"{len(self._train_dataloader)} steps per epoch")
        self._model.train()
        for epoch in range(self._args.num_epochs):
            for step, batch in enumerate(self._train_dataloader):
                outputs = self._model(batch["input_ids"], labels=batch["input_ids"])
                loss = outputs.loss
                if step % (self._args.eval_steps // 10) == 0:
                    if self._accelerator.is_main_process:
                        self.info(f"Epoch: {epoch}, Step: {step}, Train Loss: {loss}")
                self._accelerator.backward(loss)
                self._accelerator.clip_grad_norm_(self._model.parameters(), 1.0)
                self._optimizer.step()
                self._scheduler.step()
                self._optimizer.zero_grad()
                if step % self._args.eval_steps == 0:
                    loss, perplexity = self._evaluate()
                    self._model.train()
                    self._accelerator.wait_for_everyone()
                    if self._accelerator.is_main_process:
                        self.info(f"Eval Loss: {loss}, Perplexity: {perplexity}")
                        self._accelerator.unwrap_model(self._model).save_pretrained(
                            self._output_dir
                        )
                        self._tokenizer.save_pretrained(self._output_dir)
                        # self._repo.push_to_hub(commit_message=f"", blocking=False)

    def run(self):
        self._prepare_repo()
        self._prepare_data()
        self._prepare_optimizer()
        self._accelerate()
        self._train()
