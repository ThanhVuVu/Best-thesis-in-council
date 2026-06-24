from src.models.clef_pretrained import CLEFPretrainedClassifier


def build_model(name: str, num_classes: int = 3, **kwargs):
    name = name.lower()
    if name == "clef_pretrained":
        return CLEFPretrainedClassifier(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown model: {name}")
