import argparse
from argparse import RawTextHelpFormatter
import pandas

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import warnings
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Lap Trinh Ung Dung bai tap lon - Chan doan benh su dung Decision Tree hoac Random Forest.\nNgac Anh Kiet - 21020290; Pham Le Duc Thanh - 21021637 - Nguyen Vu Minh Thanh = 21020667", formatter_class=RawTextHelpFormatter)
    parser.add_argument("--train", type=str, help="Link to file csv for training.")
    parser.add_argument("--predict", type=str, help="Link to file csv that you need the model to predict.", default = '0')

    parser.add_argument("--method", type=str, help="Decision Tree DT or Random Forest RF.", default = "DT")
    parser.add_argument("--target", type=int, help="Number of targets.", default = 1)
    parser.add_argument("--ratio", type=int, help="Numbers of training data in percentage, the rest are for testing data.", default = 80)
    parser.add_argument("--max", type = int, help="Max depth of decision tree or number of sub-trees for Random Forest.", default = -1)
    
    return parser.parse_args()

def MapToNumericalValue(headers, full_df):
    final_mapping = []
    for header in headers:
        mapping_number = 0
        mapping_arr = []
        mapping_dict = {}
        for idx in range(len(full_df)): 
            mapping_arr.append(full_df.at[idx,header])
        mapping_arr = list(set(mapping_arr))
        for ele in mapping_arr:
            try:
                ele.isdigit()
            except Exception as e:
                break
            mapping_dict[ele] = mapping_number
            mapping_number +=1

        if len(mapping_dict) > 0 or 1:
            final_mapping.append(mapping_dict)
    return final_mapping

def MapDf(numerical_mapping, headers, df):
    for i in range(len(headers)):
        if len(numerical_mapping[i]) == 0:
            continue
        df[headers[i]] = df[headers[i]].map(numerical_mapping[i])
    return df

def GetSpecificRange(train_csv, p):
    df = pandas.read_csv(train_csv)
    border = int(len(df) * p / 100)
    return border

def GetDataCsv(train_csv, border):
    df = pandas.read_csv(train_csv, skiprows = lambda x: x not in range(0, border))
    return df

def GetTestCsv(train_csv, border, length_of_csv):
    specific_range_test = range(border, length_of_csv)
    df_test = pandas.read_csv(train_csv, skiprows = lambda x: x not in specific_range_test and x != 0)
    return df_test

def GetFeaturesAndTargets(train_csv, n_target):
    df = pandas.read_csv(train_csv, nrows=0).columns.tolist()
    features = df[:len(df) - n_target]
    targets = df[len(df) - n_target:]
    return features, targets

def GetConfusionMatrix(target_truth, test_predicted_dtree):
    target_truth = np.array(target_truth)
    test_predicted_dtree = np.array(test_predicted_dtree)
    confusion_matrix = metrics.confusion_matrix(target_truth, test_predicted_dtree)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
    Accuracy = metrics.accuracy_score(target_truth, test_predicted_dtree)
    Precision = metrics.precision_score(target_truth, test_predicted_dtree,  average='micro')
    Sensitivity_recall = metrics.recall_score(target_truth, test_predicted_dtree,  average='micro')
    F1_score = metrics.f1_score(target_truth, test_predicted_dtree,  average='micro')
    return cm_display, Accuracy, Precision, Sensitivity_recall, F1_score



def PredictingTest(test_predicted_dtree, test_arr, truth_arr, border):
    with open('output.txt', 'w') as f:  
        count_false_dtree = 0
        for idx in range(len(test_arr)):
            #if np.array_equal(test_predicted_dtree[idx], target_truth[idx]):
            if (test_predicted_dtree[idx] == truth_arr[idx]).all():
                continue
            else:
                count_false_dtree +=1
                f.write("Wrong prediction found at sample {}: {}\nTruth is {} while {} is predicted.\n\n".format(border + idx + 1, test_arr[idx], truth_arr[idx],test_predicted_dtree[idx]))
                #print("Wrong prediction found at sample {}: {}\nTruth is {} while {} is predicted.\n\n".format(border + idx, test_arr[idx], test_predicted_dtree[idx], truth_arr[idx]))
    
    print('Total false predictions is ', count_false_dtree, " out of ", len(test_arr))
    print('Please check output.txt for more information!')

def GenTestArr(df_test, features, targets):
    test_arr = []
    target_truth = []
    for idx in range(len(df_test)):
        temp_arr_features = []
        temp_arr_targets = []
        for ele in features:
            temp_arr_features.append(df_test.at[idx, ele])
        test_arr.append(temp_arr_features)

        for ele in targets:
            temp_arr_targets.append([df_test.at[idx, ele]])
        target_truth.append(temp_arr_targets)
    truth_arr = []
    for i in target_truth:
        t = []
        for j in i:
            t.append(j[0])
        truth_arr.append(t)

    return test_arr, truth_arr

def FitModel(df, features, targets, test_arr, method_init):
    X = df[features]
    y = df[targets]
    method = method_init.fit(X, y)

    test_predicted = method.predict(test_arr)
    return test_predicted

def PredictModel(predicting_arr, model):
    test_predicted = model.predict(predicting_arr)
    return test_predicted


def PredictingAndGenOutput(df_input, output_predicted, headers, result_txt):
    with open(result_txt, 'w') as f:  
        f.write(str(headers))
        f.write('\n')
        idx = 1
        for input in df_input:
            f.write("Case {:5}:  {:25s} | Result: {}\n".format(idx, str(input), output_predicted[idx - 1]))
            idx +=1









def main():
    warnings.filterwarnings("ignore")
    arg = parse_args()
    
    try:
        train_csv = arg.train
        if arg.ratio == 100:
            test_ratio = 99
        else:
            test_ratio = arg.ratio
        n_targets = arg.target
        method = arg.method
        input_csv = arg.predict
        if arg.max == -1:
            flag = 0
            max_arg = 100
        else:
            flag = 1
            max_arg = arg.max
    except Exception as e:
        print("Error: ",e)
        return
    
    if method == "DT":
        if flag == 1:
            method_init = DecisionTreeClassifier(criterion="gini", max_depth= max_arg)
            model_name = "Decision_Tree.pkl"
            result_txt = "Result_Decision_Tree.txt"
        else:
            method_init = DecisionTreeClassifier(criterion="gini")
            model_name = "Decision_Tree.pkl"
            result_txt = "Result_Decision_Tree.txt"


    elif method == "RF":
        method_init = RandomForestClassifier(n_estimators= max_arg)
        model_name = "RandomForest.pkl"
        result_txt = "Result_Random_Forest.txt"

    else:
        print("Method chưa chính xác, hãy chắc chắn bạn viết in hoa toàn bộ kí tự.")
        return

    if input_csv == '0': #TRAIN
        full_df = pandas.read_csv(train_csv)
        headers = pandas.read_csv(train_csv, nrows=0).columns.tolist()

        numerical_mapping = MapToNumericalValue(headers, full_df)


        length_of_csv = len(full_df)
        border = GetSpecificRange(train_csv,test_ratio)

        df = GetDataCsv(train_csv, border)
        df = MapDf(numerical_mapping, headers, df)

        df_test = GetTestCsv(train_csv, border, length_of_csv)
        df_test = MapDf(numerical_mapping, headers, df_test)
        features, targets = GetFeaturesAndTargets(train_csv, n_targets)
        test_arr, truth_arr =   GenTestArr(df_test, features, targets)
        test_predicted = FitModel(df, features, targets, test_arr, method_init)
         
        with open(model_name, 'wb') as file:  
            pickle.dump(method_init, file)
        PredictingTest(test_predicted, test_arr, truth_arr, border)
        cm_display, Accuracy, Precision, Sensitivity_recall,  F1_score = GetConfusionMatrix(truth_arr, test_predicted)

        print("Accuracy: {}\nPrecision: {}\nSensitivity: {}\nF1 score: {}".format(Accuracy, Precision, Sensitivity_recall, F1_score))
        cm_display.plot()
        plt.show()
    else:
        with open(model_name, 'rb') as file:  
            model = pickle.load(file)
        df_input = pandas.read_csv(input_csv)
        headers_input = pandas.read_csv(input_csv, nrows=0).columns.tolist()
        numerical_mapping = MapToNumericalValue(headers_input, df_input)

        df_input = MapDf(numerical_mapping, headers_input, df_input)
        features, targets = GetFeaturesAndTargets(input_csv, n_targets - 1)
        output_arr, _    =  GenTestArr(df_input, features, targets)
        p  = PredictModel(output_arr, model)

        #output_predicted = FitModel(df_input, features, targets, output_arr, method_init)
        PredictingAndGenOutput(output_arr, p ,headers_input, result_txt)
    
    return 


if __name__ == "__main__":
    main()
