import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from geomloss import SamplesLoss

# ----- 1. Configurer l'environnement -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- 2. Définir le modèle OT -----
class OTClassifier(nn.Module):
    def __init__(self, input_dim=512, proj_dim=128, n_classes=10):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )
        self.class_prototypes = nn.Parameter(torch.randn(n_classes, 16, proj_dim))
        self.ot_loss = SamplesLoss("sinkhorn", p=2, blur=0.05)

    def forward(self, x):
        x_proj = self.projector(x)
        distances = []
        for c in range(self.class_prototypes.size(0)):
            dist = self.ot_loss(x_proj, self.class_prototypes[c])
            distances.append(dist)
        logits = -torch.stack(distances)
        return logits

# ----- 3. Charger le modèle enregistré -----
checkpoint = torch.load("ot_classifier_epoch10.pth", map_location=device)
n_classes = checkpoint['n_classes']
class_names = checkpoint['class_names']

model = OTClassifier(n_classes=n_classes).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ----- 4. Charger le ResNet18 pour extraire les caractéristiques -----
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

# ----- 5. Transformation des images -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ----- 6. Fonction pour transformer une image en distribution -----
def get_distribution(image_tensor):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        features = feature_extractor(image_tensor)
    features = features.squeeze(0).permute(1, 2, 0).reshape(-1, 512)
    return features

# ----- 7. Fonction pour prédire une image -----
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    features = get_distribution(input_tensor)
    logits = model(features)
    pred = torch.argmax(logits).item()
    print(f"✅ Image '{image_path}' classée comme : {class_names[pred]}")
    return class_names[pred]

# ----- 8. Utiliser sur ton image spécifique -----
if __name__ == "__main__":
    image_path = "/raid/home/ppdsimageseg/ktaib_ach/data/test/class_1/0000001_04527_d_0000008_objet73.jpg"
    predict_image(image_path)
