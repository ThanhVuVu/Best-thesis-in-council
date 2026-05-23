from src.models.inceptiontime1d import InceptionTime1D
from src.models.resnet1d import ResNet1D
from src.models.simple_cnn import SimpleCNN1D


def build_model(name: str, num_classes: int = 3):
    name = name.lower()
    if name == "simple_cnn":
        return SimpleCNN1D(num_classes=num_classes)
    if name == "resnet1d":
        return ResNet1D(num_classes=num_classes)
    if name == "inceptiontime1d":
        return InceptionTime1D(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")
