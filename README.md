# Contrib

Implementations of recent research prototypes / networks / datasets / losses. Please feel free to share your research.

You can share your either network, loss, dataset or whole project.
Upload your contributions to the individual folder. 

For example:

Your want to share your neural network named "MyUNet". You can upload a folder named "MyUNet" containing the python script. Strix will detect the registered network automatically. The final file structure will be like:
```bash
├── Networks
│   ├── MyUNET
│   │   ├── net.py
├── Losses
├── Datasets
├── Projects
```

---
## Register your component

Here gives a simple example of how to register your network to Strix:
```
from strix import strix_networks

@strix_networks.register("2D", "segmentation", "MyUNet")
def strix_myunet(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    act: str,
    norm: str,
    n_depth: int,
    n_group: int,
    drop_out: float,
    is_prunable: bool,
    pretrained: bool,
    pretrained_model_path: str,
    **kwargs: Any
):
    inkwargs = {}
    inkwargs["in_channels"] = in_channels
    inkwargs["num_classes"] = out_channels

    return MyUnet(**inkwargs)


```
