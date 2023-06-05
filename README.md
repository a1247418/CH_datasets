# CH_datasets
Tasks for evaluating Clever Hans mitigation approaches. The tasks are a re-implementation of what has been used in the paper [Preemptively Pruning Clever-Hans Strategies in Deep Neural Networks](https://arxiv.org/abs/2304.05727).

There are five tasks in this repository:
- `mnist-8`: A modified MNIST dataset with an artifact of three pixels on images of *8*s
- `carton-crate`: A dataset of ImageNet images of _cartons_ and _crates_ with a watermark artifact on images of cartons
- `carton-envelope`: A dataset of ImageNet images of _cartons_ and _envelopes_ with a watermark artifact on images of cartons
- `carton-packet`: A dataset of ImageNet images of _cartons_ and _packets_ with a watermark artifact on images of cartons
- `isic-1`: A dataset of ISIC images with blue patch artifact on images of the _Melanocytic nevus_ class 

The corresponding data has to be downloaded separately.

## Example usage
Examples can be found in `notebooks\demonstration.ipynb`.
```python
import torch
from matplotlib import pyplot as plt    
from scenario_examples import get_scenario

def plot_scenario(train, refine, test, color_map=None):
    splits = ["train", "refine", "test"]
    n_imgs = 7
    
    for split,data in zip(splits, [train, refine, test]):
        data_loader = torch.utils.data.DataLoader(data, batch_size=n_imgs, shuffle=True)
        images, labels = next(iter(data_loader))
        fig, axes = plt.subplots(1, len(images), figsize=(n_imgs*2, 4))

        print(f"{split} data:")
        for i, image in enumerate(images):
            np_image = image.permute(1, 2, 0).numpy()
            axes[i].imshow(np_image, cmap=color_map)
        plt.tight_layout()
        plt.show()

mnist_path = "..."

scenario_m = get_scenario("mnist-8", dataset_path=mnist_path, normalize=False)

datas = []
for split in ["train", "refine", "test"]:
    datas.append(scenario_m.get_data(split))
plot_scenario(*datas, color_map="gray")
```