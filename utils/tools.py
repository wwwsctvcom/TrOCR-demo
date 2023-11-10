import random
import torch
import numpy as np
from datasets import load_metric
from transformers import set_seed
from datetime import datetime


def seed_everything(seed: int = 42) -> None:
    if seed:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_cur_time() -> str:
    """
    return: 1970-01-01
    """
    return datetime.now().strftime("%Y-%m-%d")


def get_cur_time_sec() -> str:
    """
    return: 1970-01-01 00:00:00
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def compute_cer(processor, pred_ids, label_ids):
    # load metric
    cer_metric = load_metric("cer")

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return cer
