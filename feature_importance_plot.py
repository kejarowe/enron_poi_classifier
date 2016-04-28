import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("my_feature_list.pkl","r") as feature_in:
    feature_list = pickle.load(feature_in)
with open("trained_classifier.pkl","r") as trained_classifier_in:
    clf = pickle.load(trained_classifier_in)

features = feature_list[1:]
importances = clf.feature_importances_

feature_importances = zip(importances,features)
feature_importances.sort(reverse=True)

fig,ax  = plt.subplots()
ind = np.arange(len(feature_importances))
bar_width = 0.5
rects1=ax.bar(ind,[i[0] for i in feature_importances],bar_width,color='r')
ax.set_xticklabels([f[1] for f in feature_importances],rotation='vertical')
ax.set_xticks(ind+bar_width/2)
plt.ylabel('Gini importance for base DecisionTreeClassifer')
plt.title('AdaBoostClassifier Feature Importances')
plt.tight_layout()
plt.show()
