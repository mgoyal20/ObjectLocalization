from PIL import Image
import numpy as np
import pickle
import csv


def Normalize(image, mean, std):
    for channel in range(3):
        image[:, :, channel] = (image[:, :, channel] - mean[channel]) / std[channel]
    return image


def main():
    id_to_data = {}
    id_to_size = {}
    id_to_box = {}
    path = "./data/images/1-1.png"

    with open("./data/training.csv", 'r') as f:
        id = 1
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            try:
                image = Image.open("./data/images/" + row["image_name"]).convert('RGB')
            except:
                image = Image.open(path).convert('RGB')
            id_to_size[int(id)] = np.array(image, dtype=np.float32).shape[0:2]
            image = image.resize((224, 224))
            image = np.array(image, dtype=np.float32)
            image = image / 255
            image = Normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            id_to_data[int(id)] = image
            id += 1
    f.close()
    id_to_data = np.array(list(id_to_data.values()))
    id_to_size = np.array(list(id_to_size.values()))
    f = open("./id_to_data", "wb+")
    pickle.dump(id_to_data, f, protocol=4)
    f = open("./id_to_size", "wb+")
    pickle.dump(id_to_size, f, protocol=4)

    with open("./data/training.csv", 'r') as f:
        id = 1
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            box = np.array([int(row["x1"]), int(row["x2"]), int(row["y1"]), int(row["y2"])], dtype=np.float32)
            temp_box = np.array([int(row["x1"]), int(row["x2"]), int(row["y1"]), int(row["y2"])], dtype=np.float32)
            box[0] = temp_box[0] / id_to_size[int(id) - 1][1] * 224
            box[1] = temp_box[2] / id_to_size[int(id) - 1][0] * 224
            box[2] = (temp_box[1] - temp_box[0]) / id_to_size[int(id) - 1][1] * 224
            box[3] = (temp_box[3] - temp_box[2]) / id_to_size[int(id) - 1][0] * 224
            id_to_box[int(id)] = box
            id += 1
    f.close()
    id_to_box = np.array(list(id_to_box.values()))
    f = open("./id_to_box", "wb+")
    pickle.dump(id_to_box, f, protocol=4)


if __name__ == '__main__':
    main()