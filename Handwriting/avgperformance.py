import os
import pandas as pd
import numpy as np
path = "./csvresults/mea_75_25"

accuracy = []
sensitivity = []
specificity = []
precision = []
recall = []
for file in os.listdir(path):
    data = pd.read_table(os.path.join(path, file), sep=',', engine='python')
    print(data["test_accuracy"])
    accuracy.append(data["test_accuracy"].tolist()[-1])
    sensitivity.append(data["test_sensitivity"].tolist()[-1])
    specificity.append(data["test_specificity"].tolist()[-1])
    precision.append(data["test_precision"].tolist()[-1])
    recall.append(data["test_recall"].tolist()[-1])

print("accuracy: ",round(sum(accuracy)/len(accuracy)*100,1)," std_deviation: ", round(np.std(accuracy)*100,1))
print("sensitivity: ",round(sum(sensitivity)/len(sensitivity)*100,1)," std_deviation: ", round(np.std(sensitivity)*100,1))
print("specificity: ",round(sum(specificity)/len(specificity)*100,1)," std_deviation: ", round(np.std(specificity)*100,1))
print("precision: ",round(sum(precision)/len(precision)*100,1)," std_deviation: ", round(np.std(precision)*100,1))
print("recall: ",round(sum(recall)/len(recall)*100,1)," std_deviation: ", round(np.std(recall)*100,1))
