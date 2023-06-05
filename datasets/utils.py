import inspect
from typing import Any, Tuple, Iterable
from PIL import Image
from poisoner import Poisoner


def make_poisonable(dataset_class: type):
    """
    Modifies the given Dataset class that must be a subclass of DatasetFolder to work with Poisoners in the transfrom.
    """
    if dataset_class.__name__ == "MNIST":
        def __getitem__(self, index: int) -> Tuple[Any, Any]:
            img, target = self.data[index], int(self.targets[index])
            img = Image.fromarray(img.numpy(), mode="L")
            if self.target_transform is not None:
                target = self.target_transform(target)

            if self.transform is not None:
                if isinstance(self.transform, Iterable):
                    for t in self.transform:
                        try:
                            img = t(img)
                        except TypeError:
                            img = t(img, target)
                        #
                        #if len(inspect.signature(t.__call__).parameters) == 3:#issubclass(type(t), Poisoner) or Poisoner in type(t).__bases__:
                        #    img = t(img, target)
                        #else:
                        #    img = t(img)
                else:
                    try:
                        img = self.transform(img)
                    except TypeError:
                        img = self.transform(img, target)
                    #if issubclass(type(self.transform), Poisoner):
                    #    img = self.transform(img, target)
                    #else:
                    #    img = self.transform(img)

            return img, target
    else:
        def __getitem__(self, index: int) -> Tuple[Any, Any]:
            index = int(index)
            path, target = self.samples[index]
            sample = self.loader(path)

            if self.target_transform is not None:
                target = self.target_transform(target)
            if self.transform is not None:
                if isinstance(self.transform, Iterable):
                    for t in self.transform:
                        #if issubclass(type(t), Poisoner):
                        #    sample = t(sample, target)
                        #else:
                        #    sample = t(sample)
                        try:
                            sample = t(sample)
                        except TypeError:
                            sample = t(sample, target)
                else:
                    #if issubclass(type(self.transform), Poisoner):
                    #    sample = self.transform(sample, target)
                    #else:
                    #    sample = self.transform(sample)
                    try:
                        sample = self.transform(sample)
                    except TypeError:
                        sample = self.transform(sample, target)
            return sample, target

    return type(dataset_class.__name__, (dataset_class,), {
        '__getitem__': __getitem__,
    })
