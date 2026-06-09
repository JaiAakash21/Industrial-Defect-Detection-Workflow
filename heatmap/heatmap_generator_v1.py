import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def main(file_path):

    dataset_dir = os.path.join(file_path, "Dataset")
    test_dir = os.path.join(dataset_dir, "test")

    output_file = "heatmap.png"

    generated = False

    for view in os.listdir(test_dir):

        view_path = os.path.join(test_dir, view)

        if not os.path.isdir(view_path):
            continue

        for category in ["good", "anomaly"]:

            category_path = os.path.join(view_path, category)

            if not os.path.exists(category_path):
                continue

            files = os.listdir(category_path)

            if len(files) == 0:
                continue

            img_path = os.path.join(category_path, files[0])

            img = Image.open(img_path).convert("L")

            arr = np.array(img)

            heatmap = arr / 255.0

            plt.imshow(heatmap, cmap="jet")
            plt.colorbar()
            plt.title("Heatmap")

            plt.savefig(output_file)

            generated = True
            break

        if generated:
            break

    print("HEATMAP FILE:")
    print(output_file)

    return "SUCCESS", output_file