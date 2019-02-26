import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def main():
    with open("resultsFinal.csv", 'r') as f:
        id = 1
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            box = np.array([float(row["x1"]), float(row["x2"]), float(row["y1"]), float(row["y2"])], dtype=np.float32)
            image = Image.open("./images/" + row["image_name"]).convert('RGB')
            plt.imshow(image)
            plt.gca().add_patch(
                plt.Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2],
                              fill=False, edgecolor='green', linewidth=2, alpha=0.5))
            plt.savefig("./prediction/" + str(row["image_name"]))
            plt.cla()
            print(id)
            id += 1
    f.close()


if __name__ == '__main__':
    main()
