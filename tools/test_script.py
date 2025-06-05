import numpy as np, matplotlib.pyplot as plt
a = np.load('test_runs/eyeriss/layer0/UNIFIED_TRACE.npy')
plt.plot(a[:1000])
plt.title("Interleaved line address stream")
plt.xlabel("Access index")
plt.ylabel("Line address")
plt.show()
