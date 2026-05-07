"""
加载对应的数据集。

统一数据返回格式
train_dataset[idx] : (image, label)
image.type : tensor or Image
label.type : int
"""

from pathlib import Path

from torchvision import datasets


DATASET_ROOT = Path("./dataset")


def load_image_folder(dataset_path: Path):
    # TODO: 后续补充自定义图片文件夹数据加载逻辑
    pass


def load_data(dataset_name: str):
    """
    加载指定数据集，仅负责返回原始训练集与测试集。\n
    train_dataset[idx] : (image, label)

    :param dataset_name: 数据集名称
    :return: train_dataset, test_dataset 
    """
    dataset_name = dataset_name.strip()

    if dataset_name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(
            root=str(DATASET_ROOT),
            train=True,
            download=False,
        )
        test_dataset = datasets.FashionMNIST(
            root=str(DATASET_ROOT),
            train=False,
            download=False,
        )
    elif dataset_name == "CIFAR10":
        # 这个我们会使用，上面那个不会使用
        train_dataset = datasets.CIFAR10(
            root=str(DATASET_ROOT),
            train=True,
            download=False,
        )
        test_dataset = datasets.CIFAR10(
            root=str(DATASET_ROOT),
            train=False,
            download=False,
        )
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            "Only 'FashionMNIST' and 'CIFAR10' are supported."
        )

    return train_dataset, test_dataset


if __name__ == "__main__":
    fashion_train, fashion_test = load_data("FashionMNIST")
    cifar_train, cifar_test = load_data("CIFAR10")

    print(len(fashion_train), len(fashion_test))
    print(len(cifar_train), len(cifar_test))
