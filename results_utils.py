import csv
import matplotlib.pyplot as plt

def data_init(path):
    data = dict()
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for elem in row:
                if elem in data:
                    data[elem].append(float(row[elem]))
                else:
                    data[elem] = [float(row[elem])]
    return data
