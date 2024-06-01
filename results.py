from results_utils import data_init
import matplotlib.pyplot as plt

data1 = data_init("runs/results_config1.csv")

print(data1)

plt.plot(data1["epoch"], data1["train_loss"])
plt.plot([1,2,3], [2,4,6])
plt.show()