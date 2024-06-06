from results_utils import data_init
import matplotlib.pyplot as plt

# Same approach

data1 = data_init("runs/results_config10.csv")
data2 = data_init("runs/results_config-neq-budget5.csv")
data3 = data_init("runs/results_config-neq-budget6.csv")
data4 = data_init("runs/results_config-neq-budget7.csv")
data5 = data_init("runs/results_config-neq-budget8.csv")

plt.plot(data1["epoch"], data1["acc1"], color='r', label='100% parameters')
plt.plot(data2["epoch"], data2["acc1"], color='b', label='50% parameters')
plt.plot(data3["epoch"], data3["acc1"], color='g', label='20% parameters')
plt.plot(data4["epoch"], data4["acc1"], color='m', label='10% parameters')
plt.plot(data5["epoch"], data5["acc1"], color='c', label='5% parameters')
plt.xlabel("epochs")
plt.ylabel("top1 accuracy")
plt.title("Velocity fine-tuning")
plt.legend()
plt.show()

# Approach comparison

# data1 = data_init("runs/results_config-ec1-fullr0.csv")
# data2 = data_init("runs/results_config-ec1-semir0.csv")
# data3 = data_init("runs/results_config-ec1-prop0.csv")

# plt.plot(data1["epoch"], data1["acc1"], color='b', label='full random')
# plt.plot(data2["epoch"], data2["acc1"], color='r', label='n random matrices')
# plt.plot(data3["epoch"], data3["acc1"], color='g', label='proportional matrices')
# plt.xlabel("epochs")
# plt.ylabel("top1 accuracy")
# plt.title("5% budget training from scratch")
# plt.legend()
# plt.show()

# data1 = data_init("runs/results_config1.csv")
# data2 = data_init("runs/results_config2.csv")
# data3 = data_init("runs/results_config3.csv")
# data4 = data_init("runs/results_config4.csv")
# data5 = data_init("runs/results_config5.csv")

# mem = [max(data1["unfrozen_params"])*4/1000,  max(data2["unfrozen_params"])*4/1000, max(data3["unfrozen_params"])*4/1000, max(data4["unfrozen_params"])*4/1000, max(data5["unfrozen_params"])*4/1000]

# plt.bar(['100%', '50%', '20%', '10%', '5%'], mem, color='green')
# plt.xlabel("budget")
# plt.ylabel("memory overhead (kB)")
# plt.title("Memory")
# plt.legend()
# plt.show()