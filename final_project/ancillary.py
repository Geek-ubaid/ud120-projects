import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
data_dict.pop("TOTAL", 0)
feature1 = "total_payments"
feature2 = "salary"
features = [feature1, feature2, "poi"]
data = featureFormat(data_dict, features)

for point in data:
    f1 = point[0]
    f2 = point[1]
    poi = point[2]
    if poi:
        plt.scatter(f1, f2, color = 'r')
    else:
        plt.scatter(f1, f2)

plt.xlabel(feature1)
plt.ylabel(feature2)
plt.show()