from neural_net import NeuralNet


def load_data_set():
    path = r"..\datasets\pima.txt"
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
    for i in range(8):
        max = 0
        for d in data:
            if d[i] > max:
                max = d[i]
        for d in data:
            d[i] /= max
    return data, labels


if __name__ == "__main__":
    data, labels = load_data_set()
    nn = NeuralNet([8, 12, 8, 1], ["sigmoid", "sigmoid", "sigmoid"], "se")
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
