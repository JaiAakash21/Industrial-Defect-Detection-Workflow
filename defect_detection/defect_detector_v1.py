import os
import torch

from PIL import Image
from torchvision import models, transforms

def main(file_path):

    dataset_path = os.path.join(file_path, "Dataset")
    test_path = os.path.join(dataset_path, "test")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    model.eval()

    image_file = None

    for file in os.listdir(test_path):

        if file.lower().endswith(
            (".jpg",".jpeg",".png",".bmp")
        ):
            image_file = os.path.join(test_path,file)
            break

    if image_file is None:

        return {
            "defect_status":"NO_IMAGE_FOUND"
        }

    img = Image.open(image_file).convert("RGB")

    img = transform(img).unsqueeze(0)

    with torch.no_grad():

        output = model(img)

    predicted_class = torch.argmax(
        output,
        dim=1
    ).item()

    print("IMAGE:")
    print(os.path.basename(image_file))

    print("PREDICTED INDEX:")
    print(predicted_class)

    return {
        "defect_status":"SUCCESS"
    }