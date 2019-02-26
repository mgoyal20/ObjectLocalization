from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2
import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def Normalize(image, mean, std):
    for channel in range(3):
        image[:, :, channel] = (image[:, :, channel] - mean[channel]) / std[channel]
    return image


def main():
    id_to_data = {}
    id_to_size = {}
    id_to_box = {}
    id=1
    path = "./images/1-1.png"

    with open("./data/training.csv", 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            try:
                image = Image.open("./images/" + row["image_name"]).convert('RGB')
            except:
                image = Image.open(path).convert('RGB')
            id_to_size[int(id)] = np.array(image, dtype=np.float32).shape[0:2]
            image = image.resize((224, 224))
            image = np.array(image, dtype=np.float32)
            image = image / 255
            image = Normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            id_to_data[int(id)] = image

            box = np.array([int(row["x1"]), int(row["x2"]), int(row["y1"]), int(row["y2"])], dtype=np.float32)
            temp_box = np.array([int(row["x1"]), int(row["x2"]), int(row["y1"]), int(row["y2"])], dtype=np.float32)
            box[0] = temp_box[0] / id_to_size[int(id)][1] * 224
            box[1] = temp_box[2] / id_to_size[int(id)][0] * 224
            box[2] = (temp_box[1] - temp_box[0]) / id_to_size[int(id)][1] * 224
            box[3] = (temp_box[3] - temp_box[2]) / id_to_size[int(id)][0] * 224
            id_to_box[int(id)] = box
            id += 1

            box = np.array([[float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]], dtype=np.float32)
            image = cv2.imread("./images/" + row["image_name"])[:,:,::-1]
            transforms = Sequence([RandomScale(0.2, diff = True), RandomShear(0.4), RandomHSV(hue=50, saturation=64, brightness=64)])
            image, box = transforms(image, box)
            if not box.any():
                continue
            if not image.any():
                continue
            temp_box = np.copy(box)
            image = Image.fromarray(image)
            image.save("./augImages/" + str(row["image_name"]))
            id_to_size[int(id)] = np.array(image, dtype=np.float32).shape[0:2]
            image = image.resize((224, 224))
            image = np.array(image, dtype=np.float32)
            image = image / 255
            image = Normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            id_to_data[int(id)] = image
            temp_box[0][0] = box[0][0] / id_to_size[int(id)][1] * 224
            temp_box[0][1] = box[0][1] / id_to_size[int(id)][0] * 224
            temp_box[0][2] = (box[0][2] - box[0][0]) / id_to_size[int(id)][1] * 224
            temp_box[0][3] = (box[0][3] - box[0][1]) / id_to_size[int(id)][0] * 224
            id_to_box[int(id)] = temp_box[0]
            id += 1
            print(id)
    f.close()

    id_to_data = np.array(list(id_to_data.values()))
    id_to_size = np.array(list(id_to_size.values()))
    f = open("./id_to_data", "wb+")
    pickle.dump(id_to_data, f, protocol=4)
    f = open("./id_to_size", "wb+")
    pickle.dump(id_to_size, f, protocol=4)
    id_to_box = np.array(list(id_to_box.values()))
    f = open("./id_to_box", "wb+")
    pickle.dump(id_to_box, f, protocol=4)


if __name__ == '__main__':
    main()