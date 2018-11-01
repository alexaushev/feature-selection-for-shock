import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import f_classif
from sklearn.feature_extraction import DictVectorizer

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE

from sklearn.ensemble import RandomForestClassifier
from collections import Counter

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score, precision_recall_fscore_support

import warnings
warnings.filterwarnings("ignore")

# set directories for the datasets
full_dir = "Full dataset/"
t1t2_dir = "T1+T2 dataset/"
t1_dir = "T1 dataset/"
dirs = [t1_dir, t1t2_dir, full_dir]

# set feature selection methods
methods = ['Univariate feature selection', 'Recursive feature eleminiation', "UFS + RFE", "Random forest"]

# set the number of pairs (training set, test) to use
cases = 100

# set the number of features to show
display_features_num = 80

for df_dir in dirs:
    print( '=============================================')
    print(df_dir)
    print( '=============================================')
    agg_features = list()
    agg_num = list()
    agg_acc = list()
    agg_mcc = list()
    agg_sens = list()
    agg_spes = list()
    agg_auc =list()
    
    for method in methods:
        print(method)
        chosen_features = list()
        num_features = list()
        acc_scores = list()
        mcc_scores = list()
        auc_scores = list()
        sens_scores = list()
        spec_scores = list()
        
        for case in range(1, cases + 1):
            print '.',
            
            # load training sets from corresponding folders
            train_set =  pd.read_excel(df_dir + 'train' + str(case) + '.xlsx', sheet_name = 0)
            test_set =  pd.read_excel(df_dir + 'test' + str(case) + '.xlsx', sheet_name = 0)
            
            # create target vectors
            train_y = train_set['Result in ICU']
            test_y = test_set['Result in ICU']
            
            # remove target vectors
            train_x = train_set.drop('Result in ICU', 1)
            test_x = test_set.drop('Result in ICU', 1)

            best_auc = 0
            best_acc = 0
            best_mcc = 0
            best_sens = 0
            best_spes = 0
            best_features = list()

            vocab = DictVectorizer()
            train_x = train_x.T.to_dict().values()
            orig_dict = train_x
            train_x = vocab.fit_transform(train_x)
            test_x = test_x.T.to_dict().values()
            test_x = vocab.transform(test_x)

            # pretrain models
            if method == 'Recursive feature eleminiation':
                svc = SVC(kernel = "linear")
                support = RFE(estimator = svc, n_features_to_select = 1, step = 1)
                support = support.fit(train_x, train_y)
                ranks = np.argsort(support.ranking_)
                
            elif method == 'UFS + RFE':
                # ufs 120 for T1+T2 and Full; 80 for T1
                if df_dir == t1_dir:
                    support = SelectKBest(f_classif, k=80)
                else:
                    support = SelectKBest(f_classif, k=120)
                train_x = support.fit_transform(train_x, train_y)
                test_x = support.transform(test_x)
                
                vocab.restrict(support.get_support())
                orig_dict = vocab.inverse_transform(train_x)

                svc = SVC(kernel = "linear")
                support = RFE(estimator = svc, n_features_to_select = 1, step = 1)
                support = support.fit(train_x, train_y)
                ranks = np.argsort(support.ranking_)

            # each cycle select feature sets of different size from 2 to 60
            for features in range(2, 60):
                vocab.fit(orig_dict)

                # apply the chosen feature selection method
                if method == 'Univariate feature selection':
                    support = SelectKBest(f_classif, k = features)
                    support = support.fit(train_x, train_y)

                    train_x_case = support.transform(train_x.toarray())
                    test_x_case = support.transform(test_x.toarray())
                    v = vocab
                    v.restrict(support.get_support())   
                    
                    model = GaussianNB()
                    model.fit(train_x_case, train_y)
                    
                elif method == 'Recursive feature eleminiation' or method == 'UFS + RFE':
                    train_x_case = [x[ranks[:features]] for x in train_x.toarray()]
                    test_x_case = [x[ranks[:features]] for x in test_x.toarray()]

                    chosen_f = np.zeros(len(vocab.get_feature_names()), dtype=bool)
                    chosen_f[ranks[:features]] = True
                    v = vocab
                    v.restrict(chosen_f)  
                    
                    model = SVC(kernel = "linear")
                    model.fit(train_x_case, train_y)
                    
                elif method == 'Random forest':
                    rf = RandomForestClassifier()
                    rf = rf.fit(train_x, train_y)
                    thresh = np.sort(rf.feature_importances_)[-features]
                    if thresh == 0:
                        continue
                    support = SelectFromModel(rf, threshold=thresh, prefit=True)

                    train_x_case = support.transform(train_x.toarray())
                    test_x_case = support.transform(test_x.toarray())

                    v = vocab
                    v.restrict(support.get_support())
                    
                    model = RandomForestClassifier()
                    model.fit(train_x_case, train_y)

                # get the list of selected features
                feature_names = v.get_feature_names()

                # train and evaluate the model
                pred_y = model.predict(test_x_case)
                
                # if current model has the best AUC
                auc = roc_auc_score(test_y, pred_y, average='weighted')
                if auc > best_auc:
                    best_auc = auc
                    best_acc = accuracy_score(test_y, pred_y)
                    best_mcc = matthews_corrcoef(test_y, pred_y)
                    best_sens = precision_recall_fscore_support(test_y, pred_y)[1][0]
                    best_spes = precision_recall_fscore_support(test_y, pred_y)[1][1]
                    best_features = feature_names

            # record stastistics for the best amount of features
            best_features = [i for i in best_features if i != 'NA']
            chosen_features += best_features
            mcc_scores.append(best_mcc)
            acc_scores.append(best_acc)
            auc_scores.append(best_auc)
            sens_scores.append(best_sens)
            spec_scores.append(best_spes)
            num_features.append(len(best_features))

            
        # calculate and print featues for one method
        count_features = Counter(chosen_features)
        agg_features += chosen_features
        
        # normalize the ranks
        for key in count_features:
            count_features[key] = float(count_features[key]) / float(cases)

        # print latex table with feature ranks
        display_features = count_features.most_common(display_features_num)
        print("\nStability scores:")
        print("R: " + method + ":\tScore:")
        for rank in range(0, len(display_features)):
            print(str(rank + 1) + ": " + display_features[rank][0] + "   " + str(display_features[rank][1]))

        print ('\nNumber of features: ' + str(round(np.mean(num_features), 1)) + ' $\\pm$ ' + str(round(np.std(num_features), 1)) + \
            '\nAccuracy: ' + str(round(np.mean(acc_scores), 3)) + ' $\\pm$ ' + str(round(np.std(acc_scores), 3)) + \
            '\nMCC: ' + str(round(np.mean(mcc_scores), 3)) + ' $\\pm$ ' + str(round(np.std(mcc_scores), 3)) + \
            '\nSensitivity: ' + str(round(np.mean(sens_scores), 3)) + ' $\\pm$ ' + str(round(np.std(sens_scores), 3)) + \
            '\nSpecificity: ' + str(round(np.mean(spec_scores), 3)) + ' $\\pm$ ' + str(round(np.std(spec_scores), 3)) + \
            '\nAUC: ' + str(round(np.mean(auc_scores), 3)) + ' $\\pm$ ' + str(round(np.std(auc_scores), 3)) + '\n\n')

        agg_num += num_features
        agg_acc += acc_scores
        agg_mcc += mcc_scores
        agg_sens += sens_scores
        agg_spes += spec_scores
        agg_auc += auc_scores

    # calculate and print aggregated featues for all methods (for the dataset)
    count_features = Counter(agg_features)
    for key in count_features:
        count_features[key] = float(count_features[key]) / (float(cases) * len(methods))

    display_features = count_features.most_common(display_features_num)
    print("AGGREGATED\nStability scores:")
    print("R: " + df_dir + ":\tScore:")
    for rank in range(0, len(display_features)):
        print(str(rank + 1) + ": " + display_features[rank][0] + "   " + str(display_features[rank][1]))

    print ('\nNumber of features: ' + str(round(np.mean(num_features), 1)) + ' $\\pm$ ' + str(round(np.std(num_features), 1)) + \
            '\nAccuracy: ' + str(round(np.mean(acc_scores), 3)) + ' $\\pm$ ' + str(round(np.std(acc_scores), 3)) + \
            '\nMCC: ' + str(round(np.mean(mcc_scores), 3)) + ' $\\pm$ ' + str(round(np.std(mcc_scores), 3)) + \
            '\nSensitivity: ' + str(round(np.mean(sens_scores), 3)) + ' $\\pm$ ' + str(round(np.std(sens_scores), 3)) + \
            '\nSpecificity: ' + str(round(np.mean(spec_scores), 3)) + ' $\\pm$ ' + str(round(np.std(spec_scores), 3)) + \
            '\nAUC: ' + str(round(np.mean(auc_scores), 3)) + ' $\\pm$ ' + str(round(np.std(auc_scores), 3)) + '\n\n')
        

        
