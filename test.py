import numpy as np
import matplotlib.pyplot as plt
import keras
import csv
from train import smooth_l1_loss, my_metric
from getdata import Normalize
from PIL import Image


def main():
    plt.switch_backend('agg')
    final_list = [['image_name', 'x1', 'x2', 'y1', 'y2']]

    id_to_test_data = {}
    id_to_test_size = {}
    test_images = {}
    path = "./data/images/1-1.png"

    with open("./data/test.csv", 'r') as f:
        id = 1
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            try:
                image = Image.open("./data/images/" + row["image_name"]).convert('RGB')
            except:
                image = Image.open(path).convert('RGB')
            test_images[int(id)] = str(row["image_name"])
            id_to_test_size[int(id)] = np.array(image, dtype=np.float32).shape[0:2]
            image = image.resize((224, 224))
            image = np.array(image, dtype=np.float32)
            image = image / 255
            image = Normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            id_to_test_data[int(id)] = image
            id += 1
    f.close()
    data = np.array(list(id_to_test_data.values()))
    size = np.array(list(id_to_test_size.values()))
    images = np.array(list(test_images.values()))

    index = [i for i in range(12815)]

    keras.losses.smooth_l1_loss = smooth_l1_loss
    keras.metrics.my_metric = my_metric
    model = keras.models.load_model("./model.h5")
    result = model.predict(data[index, :, :, :])

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    j = 0
    for i in index:
        image = data[i]
        prediction = result[j]
        j += 1
        for channel in range(3):
            image[:, :, channel] = image[:, :, channel] * std[channel] + mean[channel]

        image = image * 255
        image = image.astype(np.uint8)
        plt.imshow(image)
        final_list.append([images[i], prediction[0] * size[i][1], (prediction[0] + prediction[2]) * size[i][1],
                           prediction[1] * size[i][0], (prediction[1] + prediction[3]) * size[i][0]])

    with open('resultsFinal.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for row in final_list:
            writer.writerow(row)
    f.close()


if __name__ == '__main__':
    main()
