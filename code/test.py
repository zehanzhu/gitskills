import matplotlib.pyplot as plt
from matplotlib import rcParams



# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 全局字体大小设置
rcParams['font.size'] = 16  # 默认字体大小
rcParams['axes.titlesize'] = 20  # 标题字体大小
rcParams['axes.labelsize'] = 16  # 坐标轴标签字体大小
rcParams['xtick.labelsize'] = 14  # x轴刻度字体大小
rcParams['ytick.labelsize'] = 14  # y轴刻度字体大小
rcParams['legend.fontsize'] = 14  # 图例字体大小

# 示例画图
plt.plot([1, 2, 3], [4, 5, 6])
plt.title('示例折线图')  # 中文标题
plt.xlabel('横坐标')      # 中文横轴
plt.ylabel('纵坐标')      # 中文纵轴
plt.show()
