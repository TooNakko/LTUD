import argparse
from argparse import RawTextHelpFormatter
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import tree
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings


def parse_args():
    parser = argparse.ArgumentParser(description="Lap Trinh Ung Dung bai tap lon - Chan doan benh dua tren du lieu co san.\nNgac Anh Kiet - 21020290; Pham Le Duc Thanh - 21021637 - Nguyen Vu Minh Thanh = 21020", formatter_class=RawTextHelpFormatter)
    parser.add_argument("--train", type=str, required=True, help="Link to file csv for training.")
    parser.add_argument("--input", type=str, help="Link to file csv that you need the model to predict.", default = '0')
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

def GetSpecificRange(csv_link, p):
    df = pandas.read_csv(csv_link)
    border = int(len(df) * p / 100)
    return border

def GetDataCsv(csv_link, border):
    df = pandas.read_csv(csv_link, skiprows = lambda x: x not in range(0, border))
    return df

def GetTestCsv(csv_link, border, length_of_csv):
    specific_range_test = range(border, length_of_csv)
    df_test = pandas.read_csv(csv_link, skiprows = lambda x: x not in specific_range_test and x != 0)
    return df_test

def GetFeaturesAndTargets(csv_link, n_target):
    df = pandas.read_csv(csv_link, nrows=0).columns.tolist()
    features = df[:len(df) - n_target]
    targets = df[len(df) - n_target:]
    return features, targets

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

#def PredictingAndGenOutput(df_input):
#    with open('output.txt', 'w') as f:  
#        count_false_dtree = 0
#        for idx in range(len(test_arr)):
#            #if np.array_equal(test_predicted_dtree[idx], target_truth[idx]):
#            if (test_predicted_dtree[idx] == truth_arr[idx]).all():
#                continue
#            else:
#                count_false_dtree +=1
#                f.write("Wrong prediction found at sample {}: {}\nTruth is {} while {} is predicted.\n\n".format(border + idx + 1, test_arr[idx], truth_arr[idx],test_predicted_dtree[idx]))
#                #print("Wrong prediction found at sample {}: {}\nTruth is {} while {} is predicted.\n\n".format(border + idx, test_arr[idx], test_predicted_dtree[idx], truth_arr[idx]))
#    
#    print('Total false predictions is ', count_false_dtree, " out of ", len(test_arr))
#    print('Please check output.txt for more information!')
def main():
    warnings.filterwarnings("ignore")
    arg = parse_args()
    
    try:
        csv_link = arg.train
        test_ratio = arg.ratio
        n_targets = arg.target
        method = arg.method
        input_csv = arg.input
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
        else:
            method_init = DecisionTreeClassifier(criterion="gini")
    elif method == "RF":
        method_init = RandomForestClassifier(n_estimators= max_arg)
    full_df = pandas.read_csv(csv_link)
    headers = pandas.read_csv(csv_link, nrows=0).columns.tolist()
    numerical_mapping = MapToNumericalValue(headers, full_df)


    length_of_csv = len(full_df)
    border = GetSpecificRange(csv_link,test_ratio)
    
    df = GetDataCsv(csv_link, border)
    df = MapDf(numerical_mapping, headers, df)

    df_test = GetTestCsv(csv_link, border, length_of_csv)
    df_test = MapDf(numerical_mapping, headers, df_test)
    features, targets = GetFeaturesAndTargets(csv_link, n_targets)

    if input_csv != '0':
        df_input = pandas.read_csv(csv_link)
        df_input = MapDf(df_input)
        output_arr, _  =        GenTestArr(df_input, features, targets)
        output_predicted = FitModel(df, features, targets, output_arr, method_init)
        PredictingTest(test_predicted, test_arr, truth_arr, border)
    else:

        test_arr, truth_arr =   GenTestArr(df_test, features, targets)
        test_predicted = FitModel(df, features, targets, test_arr, method_init)

        PredictingTest(test_predicted, test_arr, truth_arr, border)
    
if __name__ == "__main__":
    main()
