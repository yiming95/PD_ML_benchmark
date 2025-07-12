import pandas as pd
import os
import numpy as np

path = "./output/"

accuracy = []
specificity = []
sensitivity = []
for file in os.listdir(path):
    if int(file.split("_")[2])>=8:
        break
    # 读取CSV文件
    df = pd.read_csv(path + file + "/res_pat.csv")

    # 提取Accuracy值
    accuracy.append(df['Accuracy'][0])
    specificity.append(df['Specificity'][0])
    sensitivity.append(df['Sensitivity'][0])





accuracy_std = np.std(accuracy)
specificity_std = np.std(specificity)
sensitivity_std = np.std(sensitivity)
print("Standard Deviation of Accuracy:", accuracy_std)
print("accuracy值为:",sum(accuracy) / len(accuracy))

print("specificity值为:",sum(specificity) / len(specificity))
print("Standard Deviation of Specificity:", specificity_std)

print("sensitivity值为:",sum(sensitivity) / len(sensitivity))
print("Standard Deviation of Sensitivity:", sensitivity_std)
