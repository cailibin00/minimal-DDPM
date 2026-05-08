from torch.utils.data import Dataset
from .load_data import load_data, load_image_folder
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np

"""
数据还没归一化，在送进模型的时候，要进行归一化
"""

class Dataset(Dataset):
    """
    目前兼容（使用）无条件的diffusion模型

    :param image_size: 图像尺寸
    :param is_conditional: 该数据集是否用来做有条件的diffusion模型
    :dataset_name: 数据集名称，目前支持"FashionMNIST"和"CIFAR10"
    :split: 数据集划分方式，默认为"train"，可选"train"或"test"
    """
    def __init__(self, image_size:int=32, is_conditional:bool=False,
                 dataset_name:str=None , split:str="train"):
        super().__init__()
        self.image_size = image_size
        self.dataset_name = dataset_name
        self.is_conditional = is_conditional
        self.split = split

        # 加载数据，并且打印数据集信息
        self.load_data(self.dataset_name )
        # 打印数据集信息
        self.print_info()

        # 进行数据处理
        self.transform = self.data_processor()
    
    def print_info(self):
        if self.dataset_name:
            print(f"Dataset: {self.dataset_name}")
            print(f"Dataset Split: {self.split}")
            print(f"Type : {'Conditional' if self.is_conditional else 'UnConditional'}")
            print(f"NO label will be loaded and processed." if not self.is_conditional else "Labels will be loaded and processed.")
            print("-" * 30)
            sample_image, sample_label = self.data[0]
            class_names = getattr(self.data, "classes", None)

            print(f"{self.split} Size: {len(self.data)}")
            print(f"Sample Shape: {sample_image.size}")
            print(f"Sample Label: {sample_label}")

            if class_names is not None:
                print(f"Num Classes: {len(class_names)}")
                print(f"Classes: {class_names}")

            if hasattr(self.data, "data"):
                print(f"Raw {self.split} Data Shape: {self.data.data.shape}")


    def save_example_images(self , nums :int = 5):
        """
        保存五张随机的例子图片
        """
        if not self.dataset_name:
            raise ValueError("save_example_images currently only supports dataset_name datasets.")

        save_root = Path(f"./results/dataset_examples/{self.dataset_name}/{self.split}")
        save_root.mkdir(parents=True, exist_ok=True)

        train_dataset = self.data
        total_num = len(train_dataset)
        save_num = min(nums, total_num)
        random_indices = random.sample(range(total_num), save_num)

        for i, idx in enumerate(random_indices):
            image, label = train_dataset[idx]
            image = np.asarray(image)

            save_path = save_root / f"{self.dataset_name}_{idx}_label_{label}_{i}.png"
            if image.ndim == 2:
                plt.imsave(save_path, image, cmap="gray")
            elif image.ndim == 3:
                plt.imsave(save_path, image)
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")

    def load_data(self , dataset_name:str):
        self.data = load_data(dataset_name)
        if self.split == "train":
            self.data = self.data[0]
        elif self.split == "test":
            self.data = self.data[1]
        else:
            raise ValueError(f"Unsupported split: {self.split}")

    def data_processor(self):
        """
        进行处理
        1. transform
        2. 对于标签的统一处理
        """
        import torchvision.transforms as T
        # 数据处理手段
        transform = T.Compose([
            T.Resize(self.image_size),
            T.RandomHorizontalFlip(p=0.2) ,
            T.ToTensor()
        ])

        return transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = self.transform(image)
        
        if self.is_conditional:
            return image, label
        return image
        

if __name__ == "__main__":
    dataset = Dataset(
        image_size = 32 ,
        dataset_name = "CIFAR10",
        is_conditional = True,
        split= "test"
    )
    print(len(dataset))
    print(dataset[0][1])
    dataset.save_example_images(2)
