import os
import torch
import numpy as np

from PIL import Image
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

    model.fc = torch.nn.Identity()
    model.eval()

    X = []
    y = []

    views = ["bottom","side","top"]

    for view in views:

        view_path = os.path.join(test_path, view)

        if not os.path.exists(view_path):
            continue

        for folder_name, label in [
            ("good", 0),
            ("anomaly", 1)
        ]:

            folder = os.path.join(
                view_path,
                folder_name
            )

            if not os.path.exists(folder):
                continue

            print("PROCESSING:")
            print(view, folder_name)

            for img_name in os.listdir(folder):

                try:

                    img_path = os.path.join(
                        folder,
                        img_name
                    )

                    img = Image.open(
                        img_path
                    ).convert("RGB")

                    img = transform(
                        img
                    ).unsqueeze(0)

                    with torch.no_grad():
                        feature = model(img)

                    X.append(
                        feature.squeeze().numpy()
                    )

                    y.append(label)

                except Exception as e:

                    print("ERROR:")
                    print(img_name)

    X = np.array(X)
    y = np.array(y)

    print("TOTAL SAMPLES:")
    print(len(X))

    print("FEATURE SHAPE:")
    print(X.shape)

    print("LABEL SHAPE:")
    print(y.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    clf.fit(X_train, y_train)

    preds = clf.predict(X_val)

    acc = accuracy_score(y_val, preds)

    print("ANOMALY ACCURACY:")
    print(acc)

    print("0 = GOOD")
    print("1 = ANOMALY")

    return {
        "anomaly_status":"SUCCESS"
    }