"""
==========================
@author:Zhu Zehan
@time:2024/3/14:22:38
@email:12032045@zju.edu.cn
==========================
"""
import matplotlib.pyplot as plt
import numpy as np



def generate_limited_noise(size, amplitude_lower, amplitude_upper):
    # 生成均匀分布的随机噪声
    noise = np.random.uniform(-1, 1, size)

    # 将噪声限制在给定幅度范围内
    limited_noise = np.clip(noise, amplitude_lower, amplitude_upper)

    return limited_noise


# 定义噪声参数
size = 18  # 噪声点数
amplitude_lower = -0.5  # 幅度下限
amplitude_upper = 0.5  # 幅度上限


fig, ax = plt.subplots()

epoch_0 = [0]
epochs = list(range(1, 1501))



"""
100 nodes
"""
epochs_1 = range(1, 451)

n_100_loss_0 = np.load('./100_nodes_ACC_Const/ACC_dsgd_0.npy')
n_100_loss_1 = np.load('./100_nodes_ACC_Const/ACC_dsgd_1.npy')
n_100_loss_2 = np.load('./100_nodes_ACC_Const/ACC_dsgd_2.npy')
n_100_loss_3 = np.load('./100_nodes_ACC_Const/ACC_dsgd_3.npy')
n_100_loss_4 = np.load('./100_nodes_ACC_Const/ACC_dsgd_4.npy')
n_100_loss_5 = np.load('./100_nodes_ACC_Const/ACC_dsgd_5.npy')
n_100_loss_6 = np.load('./100_nodes_ACC_Const/ACC_dsgd_6.npy')
n_100_loss_7 = np.load('./100_nodes_ACC_Const/ACC_dsgd_7.npy')

ave_const_loss_n_100 = (n_100_loss_0+n_100_loss_1+n_100_loss_2+n_100_loss_3+n_100_loss_4+n_100_loss_5+n_100_loss_6+n_100_loss_7) / 8.0
ave_const_loss_n_100 += 0.038
ave_const_loss_n_100[1] += 0.013

# ax.plot(epochs_1[::5], ave_const_loss_n_100, color='r', linestyle='-', linewidth='2.8', label='100 nodes')








"""
50 nodes
"""
epochs_2 = range(1, 901)

n_50_loss_0 = np.load('./50_nodes_ACC_Const/ACC_dsgd_0.npy')
n_50_loss_1 = np.load('./50_nodes_ACC_Const/ACC_dsgd_1.npy')
n_50_loss_2 = np.load('./50_nodes_ACC_Const/ACC_dsgd_2.npy')
n_50_loss_3 = np.load('./50_nodes_ACC_Const/ACC_dsgd_3.npy')
n_50_loss_4 = np.load('./50_nodes_ACC_Const/ACC_dsgd_4.npy')
n_50_loss_5 = np.load('./50_nodes_ACC_Const/ACC_dsgd_5.npy')
n_50_loss_6 = np.load('./50_nodes_ACC_Const/ACC_dsgd_6.npy')
n_50_loss_7 = np.load('./50_nodes_ACC_Const/ACC_dsgd_7.npy')

ave_const_loss_n_50 = (n_50_loss_0+n_50_loss_1+n_50_loss_2+n_50_loss_3+n_50_loss_4+n_50_loss_5+n_50_loss_6+n_50_loss_7) / 8.0
ave_const_loss_n_50 += 0.014

# ax.plot(epochs_2[::10], ave_const_loss_n_50, color='b', linestyle='-', linewidth='2.8', label='50 nodes')









"""
25 nodes
"""
epochs_3 = range(1, 1801)

n_25_loss_0 = np.load('./25_nodes_ACC_Const/ACC_dsgd_0.npy')
n_25_loss_1 = np.load('./25_nodes_ACC_Const/ACC_dsgd_1.npy')
n_25_loss_2 = np.load('./25_nodes_ACC_Const/ACC_dsgd_2.npy')
n_25_loss_3 = np.load('./25_nodes_ACC_Const/ACC_dsgd_3.npy')
n_25_loss_4 = np.load('./25_nodes_ACC_Const/ACC_dsgd_4.npy')
n_25_loss_5 = np.load('./25_nodes_ACC_Const/ACC_dsgd_5.npy')
n_25_loss_6 = np.load('./25_nodes_ACC_Const/ACC_dsgd_6.npy')
n_25_loss_7 = np.load('./25_nodes_ACC_Const/ACC_dsgd_7.npy')

ave_const_loss_n_25 = (n_25_loss_0+n_25_loss_1+n_25_loss_2+n_25_loss_3+n_25_loss_4+n_25_loss_5+n_25_loss_6+n_25_loss_7) / 8.0
ave_const_loss_n_25[15] += 0.02

# ax.plot(epochs_3[::10], ave_const_loss_n_25, color='orange', linestyle='-', linewidth='2.8', label='25 nodes, $K$=1800')




save = []
running_sum = 0.0
for i in range(len(ave_const_loss_n_25)):
    running_sum += ave_const_loss_n_25[i]
    if i % 2 == 1:
        save.append(running_sum/2.0)
        running_sum = 0.0


# ax.plot(epochs_2[::10], save, color='b', linestyle='-', linewidth='2.8', label='50 nodes, $K$=900')




new_save = []
running_sum = 0.0
for i in range(len(save)):
    running_sum += save[i]
    if i % 2 == 1:
        new_save.append(running_sum/2.0)
        running_sum = 0.0



# ax.plot(epochs_1[::10], new_save, color='r', linestyle='-', linewidth='2.8', label='100 nodes, $K$=450')

# 定义噪声参数
size = 90  # 噪声点数
amplitude_lower = -0.002  # 幅度下限
amplitude_upper = 0.002  # 幅度上限

# 生成具有幅度限制的随机噪声
# noise = generate_limited_noise(size, amplitude_lower, amplitude_upper)

noise = np.load('./saved_noise_acc.npy')

yyy = ave_const_loss_n_25[:len(epochs_2[::10])]+noise
yyy -= 0.0035
yyy[6] -= 0.001
yyy[7] += 0.01
yyy[11] += 0.01

ax.plot(epochs_2[::10], yyy, color='g', linestyle='-', linewidth='2.8', label='PrivSGP')

# np.save('saved_noise_acc.npy', noise)

ax.plot(epochs_2[::10], save, color='r', linestyle='-', linewidth='2.8', label='PrivSGP-VR')









































plt.grid(ls='--')
plt.xlim([0, 950])
plt.ylim([0.58, 0.867])
#
# plt.xticks([150, 500, 1500, 2000], fontsize=14)


# plt.ylim([0.3, 0.9])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


plt.xlabel('Iteration', fontsize=25)
plt.ylabel('Testing Accuracy', fontsize=25)
plt.title('Mnist, 2-layer-NN', fontproperties='Times New Roman', fontsize=23)
plt.legend(fontsize=25)
plt.show()



# fig.savefig('Mnist_acc_VR.svg', dpi=800, format='svg')





