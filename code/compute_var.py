import numpy as np

#Davis dataset
DeepGS_CI1 = [0.878, 0.870, 0.873, 0.869, 0.874]
DeepGS_rm1 = []
DeepGS_AUPR1 = []

DeepCPI_CI1 = [0.855, 0.853, 0.858, 0.867, 0.863]
DeepCPI_rm1 = []
DeepCPI_AUPR1 = []
#CI = [0.871(0.0008), 0.872(0.002), 0.878(0.004) 0.876(0.006), 0.882(0.003)]
#rm2 = [0.407(0.005), 0.644(0.006), 0.630(0.017)]
#AUPR = [0.661(0.010), 0.709(0.008), 0.714(0.010)]
arr_avg = np.mean(DeepGS_CI1)
arr_var = np.var(DeepGS_CI1)
arr_std = np.std(DeepGS_CI1, ddof=1)
print('arr: ' % DeepGS_CI1)
print('arr_avg: %f' % arr_avg)
print('arr_var: %f' % arr_var)
print('arr_std: %f' % arr_std)

arr_avg = np.mean(DeepCPI_CI1)
arr_var = np.var(DeepCPI_CI1)
arr_std = np.std(DeepCPI_CI1, ddof=1)
print('arr: ' % DeepCPI_CI1)
print('arr_avg: %f' % arr_avg)
print('arr_var: %f' % arr_var)
print('arr_std: %f' % arr_std)

#KIBA dataset

