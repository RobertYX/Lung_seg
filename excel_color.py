import pandas as pd

# 读取 csv 文件
df = pd.read_csv(r'H:\TDFPN_1124\preprocessing\prep_results\out\nodule1504.csv')

# 遍历第一列的值
for i in range(len(df)):
    if df.iloc[i, 7] == '146' or df.iloc[i, 7] == '100':
        # 如果第一列的值为 6，将对应格子标红
        df.iloc[i, 7] = 'Sign:[' + str(df.iloc[i, 7]) + ']'

# 保存修改后的文件
df.to_csv('example_updated.csv', index=False)