from src.models.catnet1d import CATNet1D
from src.models.catnet_biclassifier import CATNetBiClassifier
from src.models.catnet_rr1d import CATNetRR1D
from src.models.ecgfm_leadbridge import ECGFMLeadBridgeClassifier
from src.models.inceptiontime1d import InceptionTime1D
from src.models.macnn_se import MACNN_SE
from src.models.resnet1d import ResNet1D
from src.models.simple_cnn import SimpleCNN1D


def build_model(name: str, num_classes: int = 3, **kwargs):
    name = name.lower()
    if name == "simple_cnn":
        return SimpleCNN1D(num_classes=num_classes)
    if name == "resnet1d":
        return ResNet1D(num_classes=num_classes)
    if name == "inceptiontime1d":
        return InceptionTime1D(num_classes=num_classes)
    if name == "catnet1d":
        return CATNet1D(num_classes=num_classes, **kwargs)
    if name in {"catnet_biclassifier", "catnet_bi"}:
        return CATNetBiClassifier(num_classes=num_classes, **kwargs)
    if name == "catnet_rr1d":
        return CATNetRR1D(num_classes=num_classes, **kwargs)
    if name == "ecgfm_leadbridge":
        return ECGFMLeadBridgeClassifier(num_classes=num_classes, **kwargs)
    if name == "ecgfm_repeatbridge":
        return ECGFMLeadBridgeClassifier(num_classes=num_classes, bridge_mode="repeat", **kwargs)
    if name == "ecgfm_repeatinitbridge":
        return ECGFMLeadBridgeClassifier(num_classes=num_classes, bridge_mode="repeat_init_trainable", **kwargs)
    if name in {"macnn_se", "macnn"}:
        return MACNN_SE(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown model: {name}")
