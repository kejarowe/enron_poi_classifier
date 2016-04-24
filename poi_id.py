#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
#useful websites:
# http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif
# https://en.wikipedia.org/wiki/F-test
# http://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection.html
# http://matplotlib.org/1.4.1/examples/pylab_examples/barchart_demo.html

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features
#features_list contains all features in order of pval from ANOVA F-value (except email_address)
#try using all values first (except email address), then reduce features in reverse order if there are problems
features_list =['poi','shared_receipt_with_poi', 'from_poi_to_this_person', 'loan_advances', 'from_this_person_to_poi', 'to_messages', 'director_fees', 'total_payments', 'deferral_payments', 'exercised_stock_options', 'deferred_income', 'total_stock_value', 'from_messages', 'bonus', 'other', 'restricted_stock', 'long_term_incentive', 'expenses', 'restricted_stock_deferred', 'salary']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# to determine if I had any invalid people in my dataset, I extracted the list of people from enron61702insiderpay.pdf and compared
# that to the keys in data_dict. I found that I had 144 valid people in the data set, but 146 keys in the data dict dictionary
# the keys that I did not expect to find in data_dict were "TOTAL" and "THE TRAVEL AGENCY IN THE PARK" and since they were not valid people, I removed them
# I found wrong values for various fields in the data_dict  
del data_dict['TOTAL']
del data_dict['THE TRAVEL AGENCY IN THE PARK']
belfer_robert_entry = data_dict['BELFER ROBERT']
belfer_robert_entry['deferral_payments'] = 'NaN' #float('NaN') 
belfer_robert_entry['total_payments'] = 3285
belfer_robert_entry['exercised_stock_options'] = 'NaN' #float('NaN')
belfer_robert_entry['restricted_stock'] = 44093
belfer_robert_entry['restricted_stock_deferred'] = -44093 
belfer_robert_entry['total_stock_value'] = 'NaN' #float('NaN')
belfer_robert_entry['expenses'] = 3285
belfer_robert_entry['director_fees'] = 102500
belfer_robert_entry['deferred_income'] = -102500

bhatnagar_sanjay_entry = data_dict['BHATNAGAR SANJAY']
bhatnagar_sanjay_entry['total_payments'] = 137864
bhatnagar_sanjay_entry['exercised_stock_options'] = 15456290 
bhatnagar_sanjay_entry['restricted_stock'] = 2604490
bhatnagar_sanjay_entry['restricted_stock_deferred'] = -2604490
bhatnagar_sanjay_entry['total_stock_value'] = 15456290
bhatnagar_sanjay_entry['expenses'] = 137864
bhatnagar_sanjay_entry['director_fees'] = 'NaN' #float('NaN')
bhatnagar_sanjay_entry['other'] = 'NaN' #float('NaN')


### Task 3: Create new feature(s)
# ratio of mail to/from poi
"""
for k in data_dict:
    this_person = data_dict[k]
    total_poi_mail = float(this_person['from_this_person_to_poi']) + float(this_person['from_poi_to_this_person']) + float(this_person['shared_receipt_with_poi'])
    total_mail = float(this_person['from_messages']) + float(this_person['to_messages'])
    if total_mail > 0 and total_poi_mail > 0:
        this_person['poi_mail_ratio'] = total_poi_mail / total_mail
    else:
        this_person['poi_mail_ratio'] = 0
    print this_person['poi_mail_ratio']
"""
#features_list.append('poi_mail_ratio')

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#feature scaling
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#features = scaler.fit_transform(features,labels)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(random_state=13,n_estimators=75,learning_rate=0.5)
#clf = DecisionTreeClassifier(random_state=13)
#clf.fit(features_test,labels_test) #feature scaling does not effect Decision Tree
#clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
