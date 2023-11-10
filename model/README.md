download model from modelscope
```
from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download("microsoft/trocr-base-stage1", cache_dir='./trocr-base-stage1', revision='master')
```