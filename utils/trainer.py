import torch
from utils.tools import get_lr
from utils.tools import compute_cer
from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator


class Trainer:

    def __init__(self,
                 args=None,
                 model=None,
                 processor=None,
                 optimizer=None,
                 scheduler=None,
                 accelerator=None,
                 ):
        self.args = args
        if self.args is None:
            raise ValueError("args is None!")

        # load model, optimizer
        self.model = model
        self.processor = processor
        self.optimizer = optimizer
        if optimizer is None:
            raise ValueError("optimizer is None!")

        # load scheduler and accelerator
        self.scheduler = scheduler
        self.accelerator = accelerator if accelerator is not None else Accelerator()

    def train(self, train_data_loader=None, test_data_loader=None):
        train_data_loader, test_data_loader, self.model, self.optimizer = self.accelerator.prepare(train_data_loader,
                                                                                                   test_data_loader,
                                                                                                   self.model,
                                                                                                   self.optimizer)

        for epoch in range(1, self.args.epochs + 1):
            train_total_loss = 0
            with tqdm(enumerate(train_data_loader), total=len(train_data_loader),
                      desc=f'Epoch: {epoch}/{self.args.epochs}', postfix=dict) as train_pbar:
                self.model.train()
                for step, batch in train_pbar:
                    # backward, calculate gradient
                    with self.accelerator.autocast():
                        # forward
                        outputs = self.model(**batch)
                        loss = outputs.loss
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                    # zero the gradient
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # lr scheduler
                    if self.scheduler is not None:
                        self.scheduler.step()

                    train_total_loss += self.accelerator.gather(loss).item()

                    # update pbar
                    train_pbar.set_postfix(**{"lr": get_lr(self.optimizer),
                                              "train average loss": train_total_loss / (step + 1),
                                              "train loss": loss.item()})

            # test
            if test_data_loader is not None:
                test_total_loss = 0
                valid_cer = 0.0
                with tqdm(enumerate(test_data_loader), total=len(test_data_loader),
                          desc=f'Epoch: {epoch}/{self.args.epochs}', postfix=dict) as test_pbar:
                    self.model.eval()
                    for step, batch in test_pbar:
                        outputs = self.model(**batch)
                        loss = outputs.loss

                        pred_ids = self.model.generate(batch["pixel_values"])
                        cer = compute_cer(self.processor, pred_ids=pred_ids, label_ids=batch["labels"])
                        valid_cer += cer

                        # tqdm
                        test_total_loss += loss.item()
                        test_pbar.set_postfix(**{'test average loss': test_total_loss / (step + 1),
                                                 'test cer': valid_cer / (step + 1)})

    def save_model(self, out_dir: str = None):
        if not Path(out_dir).exists():
            Path(out_dir).mkdir()

        # unwarp model
        self.accelerator.wait_for_everyone()
        self.model = self.accelerator.unwrap_model(self.model)

        # save model
        self.model.save_pretrained(out_dir, torch_dtype=torch.float16)
        self.processor.save_pretrained(out_dir)