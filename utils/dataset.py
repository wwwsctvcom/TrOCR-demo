import torch
from torch.utils.data import Dataset


class FuDanOCRDataset(Dataset):
    """
    https://github.com/FudanVI/benchmarking-chinese-text-recognition
    """

    def __init__(self,
                 ds,
                 processor,
                 max_length=128):
        self.ds = ds
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        # get image_id, label, image(PIL.PngImagePlugin.PngImageFile image mode=RGB)
        image_id = self.ds[index]["image_id"]
        label = self.ds[index]["label"]
        image = self.ds[index]["image"]

        # prepare image (i.e. resize + normalize)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        input_ids = self.processor.tokenizer(label, padding="max_length", max_length=self.max_length).input_ids

        # important: make sure that PAD tokens are ignored by the loss function
        input_ids = [input_id if input_id != self.processor.tokenizer.pad_token_id else -100 for input_id in input_ids]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(input_ids)}
        return encoding