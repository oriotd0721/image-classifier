import torch, torchvision
from torchvision import transforms, datasets, models
from torch import nn, optim
from pathlib import Path
from tqdm import tqdm

DATA = Path("data/cats_vs_dogs")
MODEL = Path("models/resnet18.pt")
BATCH = 32
EPOCHS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def loaders(root):
    tf_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    train_ds = datasets.ImageFolder(root / "train", tf_train)
    val_ds   = datasets.ImageFolder(root / "val", tf_val)
    return (
        torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2),
        torch.utils.data.DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2),
        len(train_ds.classes)
    )

def main():
    train_loader, val_loader, num_classes = loaders(DATA)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in model.parameters(): p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)
    opt = optim.Adam(model.fc.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        for x,y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out,y)
            loss.backward(); opt.step()

        # quick val
        model.eval(); correct=total=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).argmax(1)
                correct += (pred==y).sum().item()
                total += y.size(0)
        print(f"Val accuracy: {correct/total:.3f}")

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL)
    print(f"Saved to {MODEL}")

if __name__ == "__main__":
    main()


    