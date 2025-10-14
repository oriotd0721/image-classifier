import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms, models
from torch import nn


def load_labels(labels_path: Path):
    with open(labels_path, "r") as f:
        return json.load(f)


def build_model(num_classes: int, weights_path: Path):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Inference for transfer learning classifier")
    parser.add_argument("--weights", type=str, default="models/resnet18.pt", help="path to model weights")
    parser.add_argument("--labels", type=str, default="models/labels.json", help="path to labels.json")
    parser.add_argument("image", type=str, help="path to image")
    args = parser.parse_args()

    labels = load_labels(Path(args.labels))
    model  = build_model(num_classes=len(labels), weights_path=Path(args.weights))

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])

    img = Image.open(args.image).convert("RGB")
    x = tf(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = logits.softmax(dim=1)[0]
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())

    print(f"Predicted: {labels[idx]} (p={conf:.2f})")


if __name__ == "__main__":
    main()