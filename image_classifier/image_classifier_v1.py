import os
import joblib
import torch

from PIL import Image
from torchvision import models, transforms

def main(file_path):

    print("STARTING IMAGE CLASSIFICATION")

    model_file = os.path.join(
        file_path,
        "rf_model.joblib"
    )

    if not os.path.exists(model_file):

        print("MODEL NOT FOUND")

        return "FAILED"

    clf = joblib.load(model_file)

    feature_model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    feature_model.fc = torch.nn.Identity()
    feature_model.eval()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])

    dataset_path = os.path.join(
        file_path,
        "Dataset",
        "test"
    )

    image_path = None

    for view in os.listdir(dataset_path):

        view_path = os.path.join(dataset_path, view)

        if not os.path.isdir(view_path):
            continue

        for category in ["good","anomaly"]:

            category_path = os.path.join(
                view_path,
                category
            )

            if not os.path.exists(category_path):
                continue

            files = os.listdir(category_path)

            if len(files) > 0:

                image_path = os.path.join(
                    category_path,
                    files[0]
                )

                break

        if image_path:
            break

    if image_path is None:

        print("NO IMAGE FOUND")

        return "FAILED"

    image = Image.open(image_path).convert("RGB")

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        feature = feature_model(tensor)

    feature = feature.squeeze().numpy().reshape(1,-1)

    prediction = clf.predict(feature)[0]

    label_map = {
        0: "bottom",
        1: "side",
        2: "top"
    }

    predicted_class = label_map[int(prediction)]

    print("PREDICTED CLASS:")
    print(predicted_class)

    return "SUCCESS"