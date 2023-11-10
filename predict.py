import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from modelscope.msdatasets import MsDataset


if __name__ == "__main__":
    # load dataset
    test_ds = MsDataset.load("ocr_fudanvi_zh", subset_name='scene', namespace="modelscope", split="validation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model and processor
    model = VisionEncoderDecoderModel.from_pretrained("trocr-finetuned").to(device)
    processor = TrOCRProcessor.from_pretrained("trocr-finetuned")

    # start predict
    image = test_ds[0]["image"]
    pixel_values = processor(image, return_tensors="pt")["pixel_values"].to(device)

    pred_ids = model.generate(pixel_values).squeeze()
    print(pred_ids)
    pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.decode(pred_ids, skip_special_tokens=True)
    print(pred_str)
