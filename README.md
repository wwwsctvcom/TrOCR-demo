# TrOCR-demo
TrOCR model for training and predict, offering a easy usage and easy reading demo code.


# Dataset
数据使用复旦大学OCR中文数据集：`https://modelscope.cn/datasets/modelscope/ocr_fudanvi_zh/summary`
数据集中包含scene场景的train、val和test, 数据集的调用方式如下：
```
from modelscope.msdatasets import MsDataset
ds = MsDataset.load("ocr_fudanvi_zh", subset_name='scene', namespace="modelscope", split="train")
print(ds[0])
```
数据中包含如下内容：
> {'image_id': 'image-000038361', 'label': '麻潮', 'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=101x28 at 0x7E7FD0FD6350>}

# Training
使用默认的参数开始训练；
```
python train.py
```

训练过程如下，由于数据量大，GPU计算能力不足，所以仅仅训练了少量epoch，但是模型具备初步的识别能力，如果需要商业化使用，请使用更多的数据进行训练；
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.161.03   Driver Version: 470.161.03   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  Off  | 00000000:00:04.0 Off |                    0 |
| N/A   36C    P0    26W / 250W |      2MiB / 16280MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
训练损失，学习率、cer值的变化过程如下所示：（训练速度较慢，建议可以使用lora进行微调），如果时间和算力充裕可以使用全参数训练；
```
Epoch: 1/2:   1%|          | 152/12730 [19:55<29:39:39,  8.49s/it, lr=2e-5, train average loss=3.87, train loss=3.62]
```


# Predict
使用默认的参数进行predict
```
python predict.py
```

# Cer
提供一段计算cer的示例代码，用户可以根据需求在代码中进行修改，但是需要提前初始化processor，或者直接使用；
```
import numpy as np

pred_ids = torch.tensor([1, 2, 3, -100, -100]).unsqueeze(0)
pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

label_ids = torch.tensor([1, 2, 4, -100, -100]).unsqueeze(0)
label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

cer = cer_metric.compute(predictions=pred_str, references=label_str)
print(cer)  # 1.0
```
