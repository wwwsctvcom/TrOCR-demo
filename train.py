import torch
from utils.trainer import Trainer
from utils.tools import seed_everything
from utils.dataset import FuDanOCRDataset
from modelscope.msdatasets import MsDataset
from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor
from torch.utils.data import DataLoader



class Arguments:

    def __init__(self):
        # model name or path
        self.model_name_or_path = ""

        # train
        self.epochs = 2
        self.batch_size = 5
        self.lr = 2e-5
        self.weight_decay = 1e-4

        # dataset
        self.num_workers = 12

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    args = Arguments()

    # seed
    seed_everything()

    # loading model and processor
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    # loading dataset
    train_ds = MsDataset.load("ocr_fudanvi_zh", subset_name='scene', namespace="modelscope", split="test")
    test_ds = MsDataset.load("ocr_fudanvi_zh", subset_name='scene', namespace="modelscope", split="validation")

    train_dataset = FuDanOCRDataset(ds=train_ds, processor=processor)
    test_dataset = FuDanOCRDataset(ds=test_ds, processor=processor)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # init optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           args.epochs * len(train_dataloader),
                                                           eta_min=0,
                                                           last_epoch=-1,
                                                           verbose=False)

    # start train
    trainer = Trainer(args=args,
                      model=model,
                      processor=processor,
                      optimizer=optimizer,
                      scheduler=scheduler)

    trainer.train(train_data_loader=train_dataloader, test_data_loader=test_dataloader)

    # save model
    trainer.save_model("trocr-finetuned")
