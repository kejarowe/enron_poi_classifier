import tester
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import StratifiedShuffleSplit

def main():
    clf, dataset, feature_list = tester.load_classifier_and_data()
    data = tester.featureFormat(dataset,feature_list, sort_keys = True)
    labels, features = tester.targetFeatureSplit(data)
    pipeline = Pipeline([('kbest', SelectKBest()),('ada',AdaBoostClassifier())])
    #grid_search = GridSearchCV(pipeline,{'kbest__k':range(1,20),'ada__n_estimators':[25,50,75,100],'ada__learning_rate':[ x / 10.0 for x in range(2,20,2)]},
    #  cv=StratifiedShuffleSplit(labels,50),scoring=scorer,n_jobs=4)
    grid_search = GridSearchCV(pipeline,{'kbest__k':range(10,21),'ada__n_estimators':[75],'ada__learning_rate':[0.5],'ada__random_state':[13]},
      cv=StratifiedShuffleSplit(labels,1000,test_size=0.1,random_state=42),scoring=scorer,n_jobs=4)
    grid_search.fit(features,labels)
    print "Best Parameters: ",grid_search.best_params_," score: ",grid_search.best_score_
    #print "All Scores: ",grid_search.grid_scores_
    best_features = SelectKBest(k=grid_search.best_params_['kbest__k'])
    best_features.fit(features, labels)
    best_feature_labels = []
    for include,feature in zip(best_features.get_support(),feature_list[1:]):
        if include:
            best_feature_labels.append(feature)
    print "Best Features: ",best_feature_labels

def scorer(estimator,X,Y):
    true_negatives, false_negatives, true_positives, false_positives = 0,0,0,0
    predictions = estimator.predict(X)
    for prediction, truth in zip(predictions, Y):
        if prediction == 0 and truth == 0:
            true_negatives += 1
        elif prediction == 0 and truth == 1:
            false_negatives += 1
        elif prediction == 1 and truth == 0:
            false_positives += 1
        elif prediction == 1 and truth == 1:
            true_positives += 1
        else:
            print "Warning: Found a predicted label not == 0 or 1."
            print "All predictions should take value 0 or 1."
            print "Evaluating performance for processed predictions:"
            break
    total_predictions = true_negatives + false_negatives + false_positives + true_positives
    accuracy = 1.0*(true_positives + true_negatives)/total_predictions
    precision,recall = 0,0
    if true_positives or false_positives:
        precision = 1.0*true_positives/(true_positives+false_positives)
    if true_positives or false_negatives:
        recall = 1.0*true_positives/(true_positives+false_negatives)
    f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
    return f1

if __name__=='__main__':
    main()
