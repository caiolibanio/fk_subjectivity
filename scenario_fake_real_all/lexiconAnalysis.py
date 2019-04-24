import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # For graphical representation
import seaborn as sns # Python visualization library based on matplotlib provides a high-level interface for drawing attractive statistical graphics
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import itertools
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import cross_val_predict
from scipy import interp

import warnings
warnings.filterwarnings(action='ignore')


def loadStopWordsPT(filename):
    lines = [line.rstrip('\n').strip() for line in open(filename)]
    return lines

def clean_non_alpha(df):
    for i in range(len(df)):
        list_words = df['Comment'][i].split(" ")
        list_alpha = []
        for word in list_words:
            if(word.isalpha()):
                list_alpha.append(stem_word(word))
        df.loc[i,'Comment'] = " ".join(list_alpha)

def stem_word(word):
    sno = SnowballStemmer('portuguese')
    return sno.stem(word)


def buildTFIDFVectorizerClassifier(X_train, X_test):
    # Initialize the 'tfidf_vectorizer'
    tfidf_vectorizer = TfidfVectorizer(stop_words=loadStopWordsPT('../stop_words_noAccents_pt_plusBACKUP.txt'),
                                       min_df=0.01)

    # Fit and transform the training data
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)

    # Transform the test set

    if X_test is not None:
        tfidf_test = tfidf_vectorizer.transform(X_test)
        return tfidf_train, tfidf_test, tfidf_vectorizer

    return tfidf_train, tfidf_vectorizer


def load_tfidf_values_for_test(df_input_train, input_test, vectorizer):
    df_train = df_input_train

    clean_non_alpha(df_train)

    y_train = df_train['class']

    y_train = encode_label_to_roc(y_train)
    x_train = df_train['Comment']

    if input_test is not None:
        df_test = input_test
        clean_non_alpha(df_test)

        y_test = df_test['class']
        y_test = encode_label_to_roc(y_test)
        x_test = df_test['Comment']

        if vectorizer is None:
            X_tfidf_train, X_tfidf_test, vectorizer = buildTFIDFVectorizerClassifier(x_train, x_test)
            # print("num tfidf features:", X_tfidf_train.shape)
            return X_tfidf_train, X_tfidf_test, y_train, y_test, vectorizer
        else:
            X_tfidf_train = vectorizer.transform(x_train)
            X_tfidf_test = vectorizer.transform(x_test)
            return X_tfidf_train, X_tfidf_test, y_train, y_test, vectorizer
    else:
        if vectorizer is None:
            X_tfidf_train, vectorizer = buildTFIDFVectorizerClassifier(x_train, None)
            # print("num tfidf features:", X_tfidf_train.shape)
        else:
            #X_tfidf_train = vectorizer.transform(x_train)
            #X_tfidf_test = vectorizer.transform(x_test)
            #TODO
            raise Exception('Caso nao implementado!')
    return X_tfidf_train, y_train, vectorizer

def load_tfidf_values_for_testrep_with_vectorizer(input_test, vectorizer):
    df_test = input_test
    clean_non_alpha(df_test)
    y_test = df_test['class']
    y_test = encode_label_to_roc(y_test)
    x_test = df_test['Comment']
    X_tfidf_test = vectorizer.transform(x_test)
    return X_tfidf_test, y_test

def encode_label_to_roc(y_labels):
    dict = {'class':{'fake':1,'real':0}}      # label = column name
    y_labels_out = y_labels.replace(dict, inplace = False)
    return y_labels_out

def adjust_distribution(df, n_real, n_fake, seed):
    df_real = df.loc[df['class'] == 'real']
    df_fake = df.loc[df['class'] == 'fake']
    
    if n_fake > len(df_fake):
        print('Inconsistencia na quantidade de noticias falsas!')
    if n_real > len(df_real):
        print('Inconsistencia na quantidade de noticias reais!')
    
    if n_real > 0:
        df_real = df_real.sample(frac=1, random_state=seed)[:n_real]
    if n_fake > 0:
        df_fake = df_fake.sample(frac=1, random_state=seed)[:n_fake]
    df_final = df_real.append(df_fake)
    df_final = df_final.reset_index(drop=True)
    return df_final


def ploting_curves(plt, filename):
    # roc-curve
    plt.subplot(2, 2, 1)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=0, fontsize='small')

    plt.subplot(2, 2, 2)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=0, fontsize='small')
    # roc-curve

    # pr-curve
    plt.subplot(2, 2, 3)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc=0, fontsize='small')

    plt.subplot(2, 2, 4)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc=0, fontsize='small')
    # pr-curve
    plt.savefig("output_images/" + filename + '.eps', bbox_inches='tight')
    #plt.show()

#X_tfidf_train = None
#y_tfidf_train = None
#vectorizer = None

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)


def save_results(list_results, filename):
    with open("output_values/" + filename + '.txt', 'w') as f:
        for item in list_results:
            f.write(str(item) + "\n")

def save_average_results(string_results, filename):
    with open("output_values_average/" + filename + '.txt', 'w') as f:
        f.write(string_results)


def testrep_roc_ap_curve_metrics_LEX(clf, df_train_input, df_testset_input, rep, algo_name, plt_roc, plt_pr, filename):
    tprs = []
    aucs = []
    y_real = []
    y_proba = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 1
    aps = []
    list_precision = []
    list_recall = []
    list_f1score = []
    listTNR = []
    listTPR = []
    listFNR = []
    listFPR = []

    for i in range(rep):
        df_train = adjust_distribution(df_train_input, zero_size_train, one_size_train, i)
        df_train = encode_label_to_roc(df_train)
        X_train_lex = df_train.iloc[:, 2:7]
        Y_train_lex = df_train.iloc[:, 1]
        clf.fit(X_train_lex, Y_train_lex)

        df_testset = adjust_distribution(df_testset_input, zero_size_test, one_size_test, i)
        df_testset = encode_label_to_roc(df_testset)
        X_test = df_testset.iloc[:, 2:7]
        Y_test = df_testset.iloc[:, 1]

        if "SVM linear" in algo_name:
            prediction = clf.decision_function(X_test)
        else:
            prediction = clf.predict_proba(X_test)[:, 1]

        fpr, tpr, t = roc_curve(Y_test, prediction)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # plt.subplot(2, 1, 1)
        # plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        precision, recall, _ = precision_recall_curve(Y_test, prediction)
        y_real.append(Y_test)
        y_proba.append(prediction)
        ap = average_precision_score(Y_test, prediction)
        aps.append(ap)
        lab = 'Fold %d AP=%.4f' % (i, ap)
        # plt.subplot(2, 1, 2)
        # plt.plot(recall, precision, label=lab, lw=2, alpha=0.3)

        pred = clf.predict(X_test)
        precision, recall, fmeasure, _ = precision_recall_fscore_support(Y_test, pred, average='binary')
        list_precision.append(precision)
        list_recall.append(recall)
        list_f1score.append(fmeasure)
        TP, FP, TN, FN = perf_measure(Y_test, pred)
        TNR = TN / zero_size_test
        TPR = TP / one_size_test
        FNR = FN / one_size_test
        FPR = FP / zero_size_test
        listTNR.append(TNR)
        listTPR.append(TPR)
        listFNR.append(FNR)
        listFPR.append(FPR)

        # print_conf_matrix(Y_test, pred)
        i = i + 1

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    # plt.subplot(2, 1, 1)
    if "SVM linear" in algo_name:
        alpha = 0.3
    else:
        alpha = 1
    plt_roc.plot(mean_fpr, mean_tpr, label=r' %s Mean ROC (AUC = %0.2f ) $\pm$ %0.2f' % (algo_name, mean_auc, std_auc),
                 lw=2, alpha=alpha)

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    avg_ap = average_precision_score(y_real, y_proba)
    std_ap = np.std(aps)
    lab = ' %s Mean PRC (AP=%.2f  $\pm$ %0.2f)' % (algo_name, avg_ap, std_ap)
    # plt.subplot(2, 1, 2)
    plt_pr.plot(recall, precision, label=lab, lw=2, alpha=alpha)

    save_results(aps, filename + "_" + algo_name + "_AP")
    save_results(list_precision, filename + "_" + algo_name + "_precision")
    save_results(list_recall, filename + "_" + algo_name + "_recall")
    save_results(list_f1score, filename + "_" + algo_name + "_f1")
    save_results(listTNR, filename + "_" + algo_name + "_TNR")
    save_results(listTPR, filename + "_" + algo_name + "_TPR")
    save_results(listFNR, filename + "_" + algo_name + "_FNR")
    save_results(listFPR, filename + "_" + algo_name + "_FPR")

    return avg_ap, std_ap, np.mean(list_precision), np.std(list_precision), np.mean(list_recall), np.std(list_recall), \
           np.mean(list_f1score), np.std(list_f1score), np.mean(listTNR), np.std(listTNR), np.mean(listTPR), \
           np.std(listTPR), np.mean(listFNR), np.std(listFNR), np.mean(listFPR), np.std(listFPR)


def testrep_roc_ap_curve_metrics_TFIDF(clf, df_train_input, df_testset_input, rep, algo_name, plt_roc, plt_pr, filename):
    tprs = []
    aucs = []
    y_real = []
    y_proba = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 1
    aps = []
    list_precision = []
    list_recall = []
    list_f1score = []
    listTNR = []
    listTPR = []
    listFNR = []
    listFPR = []

    for i in range(rep):
        df_train = adjust_distribution(df_train_input, zero_size_train, one_size_train, i)
        df_train = encode_label_to_roc(df_train)
        X_tfidf_train, y_tfidf_train, vectorizer = load_tfidf_values_for_test(df_train, None, None)
        clf.fit(X_tfidf_train, y_tfidf_train)

        df_testset = adjust_distribution(df_testset_input, zero_size_test, one_size_test, i)
        df_testset = encode_label_to_roc(df_testset)
        X_test, Y_test = load_tfidf_values_for_testrep_with_vectorizer(df_testset, vectorizer)

        if "SVM linear" in algo_name:
            prediction = clf.decision_function(X_test)
        else:
            prediction = clf.predict_proba(X_test)[:, 1]

        fpr, tpr, t = roc_curve(Y_test, prediction)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # plt.subplot(2, 1, 1)
        # plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        precision, recall, _ = precision_recall_curve(Y_test, prediction)
        y_real.append(Y_test)
        y_proba.append(prediction)
        ap = average_precision_score(Y_test, prediction)
        aps.append(ap)
        lab = 'Fold %d AP=%.4f' % (i, ap)
        # plt.subplot(2, 1, 2)
        # plt.plot(recall, precision, label=lab, lw=2, alpha=0.3)

        pred = clf.predict(X_test)
        precision, recall, fmeasure, _ = precision_recall_fscore_support(Y_test, pred, average='binary')
        list_precision.append(precision)
        list_recall.append(recall)
        list_f1score.append(fmeasure)
        TP, FP, TN, FN = perf_measure(Y_test, pred)
        TNR = TN / zero_size_test
        TPR = TP / one_size_test
        FNR = FN / one_size_test
        FPR = FP / zero_size_test
        listTNR.append(TNR)
        listTPR.append(TPR)
        listFNR.append(FNR)
        listFPR.append(FPR)

        # print_conf_matrix(Y_test, pred)
        i = i + 1

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    # plt.subplot(2, 1, 1)
    if "SVM linear" in algo_name:
        alpha = 0.3
    else:
        alpha = 1
    plt_roc.plot(mean_fpr, mean_tpr, label=r' %s Mean ROC (AUC = %0.2f ) $\pm$ %0.2f' % (algo_name, mean_auc, std_auc),
                 lw=2, alpha=alpha)

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    avg_ap = average_precision_score(y_real, y_proba)
    std_ap = np.std(aps)
    lab = ' %s Mean PRC (AP=%.2f  $\pm$ %0.2f)' % (algo_name, avg_ap, std_ap)
    # plt.subplot(2, 1, 2)
    plt_pr.plot(recall, precision, label=lab, lw=2, alpha=alpha)

    save_results(aps, filename + "_" + algo_name + "_AP")
    save_results(list_precision, filename + "_" + algo_name + "_precision")
    save_results(list_recall, filename + "_" + algo_name + "_recall")
    save_results(list_f1score, filename + "_" + algo_name + "_f1")
    save_results(listTNR, filename + "_" + algo_name + "_TNR")
    save_results(listTPR, filename + "_" + algo_name + "_TPR")
    save_results(listFNR, filename + "_" + algo_name + "_FNR")
    save_results(listFPR, filename + "_" + algo_name + "_FPR")

    return avg_ap, std_ap, np.mean(list_precision), np.std(list_precision), np.mean(list_recall), np.std(list_recall),\
           np.mean(list_f1score), np.std(list_f1score), np.mean(listTNR), np.std(listTNR), np.mean(listTPR),\
           np.std(listTPR), np.mean(listFNR), np.std(listFNR), np.mean(listFPR), np.std(listFPR)


def plot_all_roc_curves_test_repetitions(df_train_input, df_testset_input, repetitions, filename):
    result = []
    plt.clf()
    fig = plt.figure(figsize=(15, 15))
    clfs = []
    clfs.append(('Dummy stratified_LEX', DummyClassifier(strategy='stratified')))
    clfs.append(('XGBClassifier_LEX', XGBClassifier(n_jobs=4)))
    clfs.append(('RandomForestClassifier_LEX', RandomForestClassifier()))
    # clfs.append(('SVM LEX', BaggingClassifier(SVC(kernel='rbf', probability=True), max_samples=1.0 / 5, n_estimators=5, random_state=seed, n_jobs=-1)))

    clfs.append(('Dummy stratified_TFIDF', DummyClassifier(strategy='stratified')))
    clfs.append(('XGBClassifier_TFIDF', XGBClassifier(n_jobs=4)))
    clfs.append(('RandomForestClassifier_TFIDF', RandomForestClassifier()))
    # clfs.append(('SVM TFIDF', BaggingClassifier(SVC(kernel='rbf', probability=True), max_samples=1.0 / 5, n_estimators=5, random_state=seed, n_jobs=-1)))

    # df_train = adjust_distribution(df_train_input, zero_size_train, one_size_train, seed)
    # df_train = encode_label_to_roc(df_train)
    # X_train_lex = df_train.iloc[:,2:7]
    # Y_train_lex = df_train.iloc[:,1]

    # X_tfidf_train, X_tfidf_test, y_tfidf_train, y_tfidf_test, train_vectorizer = load_tfidf_values_for_test(df_train, df_testset, None)
    for algo_name, clf in clfs:
        #print(algo_name + " Reporting results...")
        if 'TFIDF' in algo_name:
            clf.probability = True
            # clf.fit(X_tfidf_train, y_tfidf_train)
            plt_roc_tfidf = plt.subplot(2, 2, 2)
            plt_pr_tfidf = plt.subplot(2, 2, 4)
            mean_ap, std_ap, mean_precision, std_precision, mean_recall, std_recall, mean_f1score, std_f1score, \
            mean_TNR, std_TNR, mean_TPR, std_TPR, mean_FNR, std_FNR, mean_FPR, std_FPR = testrep_roc_ap_curve_metrics_TFIDF(
                clf, df_train_input, df_testset_input, repetitions, algo_name, plt_roc_tfidf, plt_pr_tfidf, filename)
        else:
            clf.probability = True
            # clf.fit(X_train_lex, Y_train_lex)
            plt_roc_lex = plt.subplot(2, 2, 1)
            plt_pr_lex = plt.subplot(2, 2, 3)
            mean_ap, std_ap, mean_precision, std_precision, mean_recall, std_recall, mean_f1score, std_f1score, \
            mean_TNR, std_TNR, mean_TPR, std_TPR, mean_FNR, std_FNR, mean_FPR, std_FPR = testrep_roc_ap_curve_metrics_LEX(
                clf, df_train_input, df_testset_input, repetitions, algo_name, plt_roc_lex, plt_pr_lex, filename)
        response = "%s\nMean AUC-PRC: %0.2f\nStd AUC-PRC: %0.2f\nMean Precision: %0.2f\nSTD Precision: %0.2f\n" \
                   "Mean Recall: %0.2f\nSTD Recall: %0.2f\nMean F1: %0.2f\nSTD F1: %0.2f\nMean TNR: %0.2f\nSTD TNR: %0.2f\n" \
                   "Mean TPR: %0.2f\nSTD TPR: %0.2f\nMean FNR: %0.2f\nSTD FNR: %0.2f\nMean FPR: %0.2f\nSTD FPR: %0.2f"\
                   % (algo_name, mean_ap, std_ap, mean_precision, std_precision, mean_recall, std_recall, mean_f1score, std_f1score, \
            mean_TNR, std_TNR, mean_TPR, std_TPR, mean_FNR, std_FNR, mean_FPR, std_FPR)
        #print(response)
        #save_average_results(response, filename + "_" + algo_name + "_" + "AVG-results")
        #print("------------------------------------------------------------------------------------------")

    ploting_curves(plt, filename)


def execute_test_scenarios(train_dataset_path, test_list_paths, repetitions, base_file_name):
    # Loading train set
    a = pd.read_csv(train_dataset_path)
    df_train = pd.DataFrame(a)
    df_train.drop_duplicates(subset=['Comment'], keep='first', inplace=True)
    # Loading test set
    for index, path in enumerate(test_list_paths, 1):
        a = pd.read_csv(path)
        df_testset = pd.DataFrame(a)
        plot_all_roc_curves_test_repetitions(df_train, df_testset, repetitions, base_file_name + "_" + str(index))


zero_size_train = 260
one_size_train = 65
zero_size_test = 120
one_size_test = 30


#caso 1
print("Starting case 1...")
train_dataset_path = '../../../data/scenario_fake_real_all/lexicon_rates_without_stopwords_REAL_FAKE_ALLTOPICS_ESTADAO_FOLHA_SENTENCES_ALL_TRAIN.csv'
test_list_paths = [
    '../../../data/scenario_fake_real_all/lexicon_rates_without_stopwords_REAL_FAKE_ALLTOPICS_ESTADAO_FOLHA_SENTENCES_ALL_TEST.csv'
]
repetitions = 500
base_file_name = "all-data"
execute_test_scenarios(train_dataset_path, test_list_paths, repetitions, base_file_name)
