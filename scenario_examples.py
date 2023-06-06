from itertools import chain
from typing import Optional, List

import numpy as np
import torch
from torchvision.transforms import Resize
from datasets.splits import sample_indicators
from poisoner import PastePoisoner, PixelPoisoner, Poisoner
from scenario import Scenario, get_default_transform

from utils import get_artifact_path, get_refinement_indices_path


class CartonPoisoner(PastePoisoner):
    def __init__(self, p: float, classes: Optional[List[int]] = None):
        super().__init__(
            artifact_paths=[
                get_artifact_path("carton_logo4.png"),
                get_artifact_path("alibaba_logo2.png"),
            ],
            positions=[(10, 110), (142, 208)],
            p=p,
            opacity=0.9,
            classes=classes,
        )


class MtbPoisoner(PastePoisoner):
    def __init__(
        self, p: float, classes: Optional[List[int]] = None, shrink: bool = True
    ):
        super().__init__([], [(0, 0)], p, classes)
        self.poison_before_tensor = False
        self.shrink = shrink
        self.artifact = torch.load(get_artifact_path("mb_logo"))

    def _poison(self, img: torch.Tensor) -> torch.Tensor:
        if self.shrink:
            img[:, 17:-17, 22:-21] = Resize(img[:, 17:-17, 22:-21].shape[-2:])(img)

        img[:, :17] = self.artifact[:, :17]
        img[:, -17:] = self.artifact[:, -17:]
        img[:, :, :22] = self.artifact[:, :, :22]
        img[:, :, -21:] = self.artifact[:, :, -21:]
        return img


class ImageNetScenario(Scenario):
    def __init__(
        self,
        dataset_dir: str,
        target_class: int,
        background_classes: List[int],
        poisoner_class: Poisoner,
        train_p: float = 0.0,
        refinement_p: float = 0.0,
        test_p: float = 1.0,
        normalize: bool = True,
        poisoning_stategy: str = "uniform",
    ):

        self.label_mapping = {
            c: i for i, c in enumerate([target_class] + background_classes)
        }
        target_transform = lambda x: self.label_mapping[x]
        class_subset = list(self.label_mapping.keys())

        if poisoning_stategy == "uniform":
            to_poison = np.arange(len(class_subset))
        elif poisoning_stategy == "adversarial":
            to_poison = np.arange(len(class_subset))[1:]
        else:
            raise ValueError(f"Unknown poisoning strategy: {poisoning_stategy}")
        train_poisoner = poisoner_class(p=train_p, classes=None)
        refinement_poisoner = poisoner_class(p=refinement_p, classes=None)
        test_poisoner = poisoner_class(p=test_p, classes=to_poison)

        train_idcs = list(
            chain(
                *[
                    sample_indicators["imagenet"]["train"]["all"][k]
                    for k in class_subset
                ]
            )
        )
        refinement_idcs = list(
            chain(
                *[
                    sample_indicators["imagenet"]["train"]["clean"][k]
                    for k in class_subset
                ]
            )
        )
        test_idcs = list(
            chain(
                *[
                    sample_indicators["imagenet"]["test"]["clean"][k]
                    for k in class_subset
                ]
            )
        )

        super().__init__(
            "imagenet",
            dataset_dir=dataset_dir,
            train_idcs=train_idcs,
            refinement_idcs=refinement_idcs,
            test_idcs=test_idcs,
            train_poisoner=train_poisoner,
            refinement_poisoner=refinement_poisoner,
            test_poisoner=test_poisoner,
            transform=get_default_transform("imagenet", normalize=normalize),
            target_transform=target_transform,
        )


class MNISTPoisoner(PixelPoisoner):
    def __init__(self, p: float, classes: Optional[List[int]] = None):
        super().__init__(
            pixels=[([1, 2], [2, 2]), ([2, 2], [1, 2])],
            pixel_value=1.0,
            p=p,
            classes=classes,
        )
        self.poison_before_tensor = False


class MNISTScenario(Scenario):
    def __init__(
        self,
        dataset_dir: str,
        target_class: int,
        background_classes: List[int],
        poisoner_class: Poisoner,
        train_p: float = 0.7,
        refinement_p: float = 0.0,
        test_p: float = 1.0,
        normalize: bool = True,
        poisoning_stategy: str = "uniform",
    ):

        if poisoning_stategy == "uniform":
            to_poison = [target_class] + background_classes
        elif poisoning_stategy == "adversarial":
            to_poison = background_classes
        else:
            raise ValueError(f"Unknown poisoning strategy: {poisoning_stategy}")
        train_poisoner = poisoner_class(p=train_p, classes=[target_class])
        refinement_poisoner = poisoner_class(p=refinement_p, classes=None)
        test_poisoner = poisoner_class(p=test_p, classes=to_poison)

        # Load train and test indices
        idcs_path = get_refinement_indices_path("mnist_refinement_idcs.npy")
        refinement_idcs = np.load(idcs_path)
        train_idcs = test_idcs = None

        super().__init__(
            "mnist",
            dataset_dir=dataset_dir,
            train_idcs=train_idcs,
            refinement_idcs=refinement_idcs,
            test_idcs=test_idcs,
            train_poisoner=train_poisoner,
            refinement_poisoner=refinement_poisoner,
            test_poisoner=test_poisoner,
            transform=get_default_transform("mnist", normalize=normalize),
        )


class ISICPoisoner(PastePoisoner):
    def __init__(self, p: float, classes: Optional[List[int]] = None):
        super().__init__(
            artifact_paths=get_artifact_path("blue_patch.png"),
            positions=[(0, 0)],
            opacity=1.0,
            p=p,
            classes=classes,
        )


class ISICScenario(Scenario):
    def __init__(
        self,
        dataset_dir: str,
        target_class: int,
        background_classes: List[int],
        poisoner_class: Poisoner,
        train_p: float = 0.0,
        refinement_p: float = 0.0,
        test_p: float = 1.0,
        normalize: bool = True,
        poisoning_stategy: str = "uniform",
    ):

        if poisoning_stategy == "uniform":
            to_poison = [target_class] + background_classes
        elif poisoning_stategy == "adversarial":
            to_poison = background_classes
        else:
            raise ValueError(f"Unknown poisoning strategy: {poisoning_stategy}")
        train_poisoner = poisoner_class(p=train_p, classes=None)
        refinement_poisoner = poisoner_class(p=refinement_p, classes=None)
        test_poisoner = poisoner_class(p=test_p, classes=to_poison)

        train_idcs = list(
                sample_indicators["isic"]["train"]["all"]
        )
        refinement_idcs = list(
                sample_indicators["isic"]["train"]["clean"]
        )
        test_idcs = list(
                sample_indicators["isic"]["test"]["clean"]
        )
        super().__init__(
            "isic",
            dataset_dir=dataset_dir,
            train_idcs=train_idcs,
            refinement_idcs=refinement_idcs,
            test_idcs=test_idcs,
            train_poisoner=train_poisoner,
            refinement_poisoner=refinement_poisoner,
            test_poisoner=test_poisoner,
            transform=get_default_transform("isic", normalize=normalize),
        )


def get_scenario(scenario: str, dataset_path: str, normalize: bool = True) -> Scenario:
    """Returns a scenario object for the given scenario name.
    Args:
        scenario: Name of the scenario. One of "carton-crate", "carton-envelope", "carton-packet", "mtb-bbt", "mnist-8", "isic-1".
        dataset_path: Path to the dataset.
        normalize: Whether to normalize the images.
    Returns:
        Scenario object.
    """
    if scenario == "carton-crate":
        return ImageNetScenario(
            dataset_path, 478, [519], CartonPoisoner, normalize=normalize
        )
    elif scenario == "carton-envelope":
        return ImageNetScenario(
            dataset_path, 478, [549], CartonPoisoner, normalize=normalize
        )
    elif scenario == "carton-packet":
        return ImageNetScenario(
            dataset_path, 478, [692], CartonPoisoner, normalize=normalize
        )
    elif scenario == "mtb-bbt":
        return ImageNetScenario(
            dataset_path, 671, [444], MtbPoisoner, normalize=normalize
        )
    elif scenario == "mnist-8":
        return MNISTScenario(
            dataset_path,
            8,
            [0, 1, 2, 3, 4, 5, 6, 7, 9],
            MNISTPoisoner,
            normalize=normalize,
        )
    elif scenario == "isic-1":
        return ISICScenario(
            dataset_path,
            1,
            [0, 2, 3, 4, 5, 6, 7],
            ISICPoisoner,
            normalize=normalize,
        )
    else:
        raise ValueError(f"Unknown scenario {scenario}")
