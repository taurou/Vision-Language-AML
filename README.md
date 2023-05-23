# AML 2022/2023 Project - Vision and Language
Official repository for the "Vision and Language" project - Advanced Machine Learning 2022/2023 @ PoliTo

## Getting Started
Make sure to have a CUDA capable device, supporting at least CUDA 11.1, installed and correctly configured on your system. 

(The base code of this project has been produced using CUDA 11.3 and Python 3.9.13)

Once you have properly setup everything, make sure you are in the correct directory and run from the command line:
```bash
pip install -r requirements.txt
```

### Dataset
1. Download PACS dataset from the portal of the course in the "project_topics" folder.
2. Place the dataset in the 'data/PACS' folder making sure that the images are organized in this way:
```
data/PACS/kfold/art_painting/dog/pic_001.jpg
data/PACS/kfold/art_painting/dog/pic_002.jpg
data/PACS/kfold/art_painting/dog/pic_003.jpg
...
```

At this point you should be able to run and edit the base code provided.

## Base Code Structure
| File | Description |
| ---- | ----------- |
| `main.py` | main entry point. Contains the logic needed to properly setup and run each experiment. |
| `parse_args.py` | contains the function responsible for parsing each command line argument. |
| `load_data.py` | contains the code to load data, build splits and dataloaders. |
| `models/base_model.py` | contains the architectures used in the project. |
| `experiments/baseline.py` | contains the code to reproduce the baseline experiment (see point 1. of the project) |
| `experiments/domain_disentangle.py` | contains the skeleton code to implement the domain disentanglement experiment (see point 2. of the project) |
| `experiments/clip_disentangle.py` | contains the skeleton code to implement the disentanglement experiment using CLIP (see point 4. of the project) |

## Base Command Line Arguments
| Argument &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;  | Description |
| -------- | ----------- |
| `--experiment` | which experiment to run chosen among the following: |
| | `baseline`: runs the experiment needed to reproduce the baseline (see point 1. of the project) |
| | `domain_disentangle`: runs the domain disentanglement experiment (see point 2. of the project) |
| | `clip_disentangle`: runs the disentanglement experiment using CLIP (see point 4. of the project) |

| `--target_domain` | which visual domain to use as the target domain choosing among `art_painting`, `cartoon`, `sketch`, `photo` |
| `--lr` | learning rate used in the optimization procedure. *Do not change it.* |
| `--max_iterations` | total number of iterations of the optimization procedure. *Do not change it*, unless you have to reduce batch size. In that case, (max_iterations / batch_size) ratio shall be constant. |
| `--batch_size` | batch size used in the optimization procedure. The default value should be fine for GPUs with at least 4GB of dedicated GPU memory. *Do not change it*, unless you have to reduce batch size. In that case, (max_iterations / batch_size) ratio shall be constant. |
| `--num_workers` | total number of worker processes to spawn to speed up data loading. Tweak it according to your hardware specs. |
| `--print_every` | controls how frequently the mean training loss is displayed. |
| `--validate_every` | controls how frequently the validation procedure occurs. *Do not change it.* |
| `--output_path` | points to the root directory where the _records/_ folder (containing the results of each experiments) will be created. |
| `--data_path` | points to the directory where the PACS dataset is stored. See above (Getting Started >>> Dataset) to properly setup this argument. |
| `--cpu` | if set, the experiment will run on the CPU. |
| `--weights` | if set, the experiment will run on different weights than the default value. Format: --weights=[weight1, weight2, weight3, alpha, clip_weight] |
| `--dom_gen` | if set, the experiment use domain generalization. |
| `--test` | if set, the experiment will skip the training procedure and just run the evaluation on the test set. |

## Baseline Results (see point 1. of the project)
|          | Art Painting &#8594; Cartoon | Art Painting &#8594; Sketch | Art Painting &#8594; Photo | Average |
| :------: | :--------------------------: | :-------------------------: | :------------------------: | :-----: |
| Baseline |            59.04             |             58.72           |            94.07           |  70.61  |

## CLIP Text Encoder
The following code fragment should provide an hint on how to use CLIP's text encoder.

```python
import torch
import clip

# Load CLIP model and freeze it
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

clip_model, _ = clip.load('ViT-B/32', device='cpu') # load it first to CPU to ensure you're using fp32 precision.
clip_model = clip_model.to(device)
clip_model.eval()
for param in clip_model.parameters():
    param.requires_grad = False

# To use the textual encoder:
description = 'a picture of a small dog playing with a ball'
tokenized_text = clip.tokenize(description).to(device)

text_features = clip_model.encode_text(tokenized_text)

```
