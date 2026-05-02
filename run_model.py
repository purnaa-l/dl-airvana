import torch
from torchvision import models, transforms, datasets
from PIL import Image
import sys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- CONFIG ----
MODEL_PATH = "resnet18_finetune.pt"
DATA_ROOT = "data/train"   # only used to recover class names
NUM_CLASSES = len(datasets.ImageFolder(DATA_ROOT).classes)
# ----------------

# Rebuild model architecture (MUST match training)
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),   # MUST match training DROPOUT
    torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
)

# Load weights
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# Image preprocessing (same as validation)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# Load image
img = Image.open(sys.argv[1]).convert("RGB")
x = transform(img).unsqueeze(0).to(DEVICE)

# Predict
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    pred = probs.argmax().item()

classes = datasets.ImageFolder(DATA_ROOT).classes
print("Predicted class:", classes[pred])
print("Confidence:", probs[pred].item())
