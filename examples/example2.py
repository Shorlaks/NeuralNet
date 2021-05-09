from neural_net import NeuralNet


def load_data_set():
    path = r"..\datasets\house_price.txt"
    data = []
    labels = []
    file = open(path, 'r')
    lines = file.readlines()
    for line in lines:
        arr = line.split(',')
        line_data = []
        for i in range(len(arr) - 1):
            line_data.append(float(arr[i]))
        data.append(line_data)
        labels.append([float(arr[-1][:1])])

    # Normalize
    for j in range(10):
        max = 0
        for d in data:
            if d[j] > max:
                max = d[j]
        for d in data:
            d[j] /= max
    return data, labels


if __name__ == "__main__":
    data, labels = load_data_set()
    nn = NeuralNet([10, 32, 32, 1], ["sigmoid", "sigmoid", "sigmoid"], "bce")
    nn.train(data, labels, epochs=10)
    counter = 0
    for i in range(100):
        a = nn.predict(data[i])
        if a[0] <= 0.5 and labels[i][0] == 0:
            counter += 1
        if a[0] > 0.5 and labels[i][0] == 1:
            counter += 1
    print(f'Total Accuracy: {counter / 100}')
    print("Finished")



