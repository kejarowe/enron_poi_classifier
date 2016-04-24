#analysis of available features to determine which to select
import poi_id
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
import numpy as np

#get features from dataset
enron_ppl = poi_id.data_dict.keys()
all_features = poi_id.data_dict[enron_ppl[0]].keys()
print "Number of features (does not include poi label): ", len(all_features)-1
print "Number of points/samples: ", len(enron_ppl)
poi_total, non_poi_total = 0,0
for person in enron_ppl:
    poi = poi_id.data_dict[person]['poi']
    if poi:
        poi_total = poi_total + 1
    else:
        non_poi_total = non_poi_total + 1
print "num poi: ", poi_total, " , num non-poi: ",non_poi_total
all_features.remove('email_address')
all_features.remove('poi')
all_features.insert(0,'poi') #move 'poi' to the beginning of the list so that it will be identified as the label

#split labels and features
labels, features = poi_id.targetFeatureSplit(poi_id.featureFormat(poi_id.data_dict, all_features))

#generate plot of feature p-values
F, pval = f_classif(features, labels)
feature_probs = zip(-np.log10(pval), all_features[1:])#zip for association while sorting
feature_probs.sort(reverse=True)

fig, ax = plt.subplots()
ind = np.arange(len(feature_probs))
bar_width = 0.5
rects1=ax.bar(ind,[f[0] for f in feature_probs],bar_width,color='r')
ax.set_xticklabels([f[1] for f in feature_probs],rotation='vertical')
ax.set_xticks(ind+bar_width/2)
plt.ylabel('-log10(pval)')
plt.title('f_classif pval by feature')
plt.tight_layout()
plt.show()
