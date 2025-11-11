for ndx in range(2):
    print(ndx)

import pandas as pd

# 读取Excel文件
df = pd.read_excel('nodule.xls')

ct_counts = df['CT号'].value_counts()

# 打印相同CT号的数量
print(ct_counts)



# 读取Excel文件
df = pd.read_excel('nodule.xls', sheet_name='123')

# 获取对象数量
object_count = len(df.index)

ct_counts = df['CT号'].value_counts()

# 打印相同CT号的数量
print(ct_counts)
# 打印对象数量
print("对象数量：", object_count)


df = pd.read_excel('nodule.xls', sheet_name='123')
# 提取x、y、z和直径列
x_values = df['x']
y_values = df['y']
z_values = df['z']
diameter_values = df['大小(cm)']

# 打印属性值
for i in range(len(df)):
    print(f"对象{i+1}：x={x_values[i]}, y={y_values[i]}, z={z_values[i]}, 大小(cm)={diameter_values[i]}")
