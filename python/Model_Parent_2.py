

################################################################################################
# IMPORT FUNTIONS & GLOBAL PARAMETERS
################################################################################################

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from prettytable import PrettyTable as pt

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import statsmodels.api as sm

from keras.models import Sequential
from keras.layers import Dense

# library I got aic and bic from is listed below. Underlying calculations are shown there.
# https://pypi.org/project/RegscorePy/
from RegscorePy import * 

# Setting size of all matplotlib graphs
plt.rcParams.update({'font.size': 14})


################################################################################################
# HELPER FUNCTIONS
################################################################################################


# Function to quickly show visual preformance of the model
# Plots y vs y_predicted
def showFit(x_values, y_test, y_pred):
    plt.scatter(x_values, y_test, label="y_test", color="black")
    plt.plot(x_values, y_pred, label="y_pred", color="blue")
    plt.legend()
    plt.show()
    
    
# Calculates AIC based on equation list on the follow webpage: https://pypi.org/project/RegscorePy/
def calc_aic(y, y_pred, num_features):
    
    n = len(y)
    
    sum_list = []
    for i in range (0, len(y)):
        sum_list.append((y[i] - y_pred[i])**2)
        
    rss = sum(sum_list)
    
    aic_val = n * np.log(rss/n) + 2 * num_features
    
    return aic_val


# Calculates BIC based on equation list on the follow webpage: https://pypi.org/project/RegscorePy/
def calc_bic (y, y_pred, num_features):
    
    n = len(y)
    
    sum_list = []
    for i in range (0, len(y)):
        sum_list.append((y[i] - y_pred[i])**2)
        
    rss = sum(sum_list)
    
    aic_val = n * np.log(rss/n) + num_features * np.log(n)
    
    return aic_val    


# Caluculates r2_bar base on formula outlined in pg. 157-158 of the textbook: https://cobweb.cs.uga.edu/~jam/scalation_guide/comp_data_science.pdf
def calc_r2_bar(m, n, r2):
        
    #m = number of data points
    #n = number of features
    
    dfr = n - 1
    df = m - n
    
    rdf = (dfr + df) / df           #ratio of total degrees of freedom to degrees of freedom for error 
    r2_bar = 1 - rdf * (1 - r2)
    
    return r2_bar


# Splits data into and X_train set, X_test set, y_train set, and y_test set based on KFold cross validation with n number of splits
# Default train to test ratio is 80:20
def splitData(X, y, num_splits):

    X_arr = X.to_numpy()
    y_arr = y.to_numpy()
    
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=1)
    kf.get_n_splits(X_arr, y_arr)
    
    ret_list = []

    for train_index, test_index in kf.split(X_arr, y_arr):
        X_train, X_test, y_train, y_test = X_arr[train_index], X_arr[test_index], y_arr[train_index], y_arr[test_index]
        
        ret_list.append([X_train, X_test, y_train, y_test])
        
    return ret_list


# Using OLS statmodels lib to calculate p-values for each feature
# function returns a dataframe sorted in ascending order with the coresponding feature name and index postion
# column headers are [index, feature_name, p_value]
def calc_p_values(model, X, y, bool, name):
    # Statsmodels.OLS requires us to add a constant.
    x = sm.add_constant(X)
    model = sm.OLS(y , x)
    results = model.fit()
    
    print()
    print(name + " SELECTION REPORT:")
    print()
    print(results.summary())
    
    df_pval = pd.DataFrame()
    df_pval['pval'] = results.pvalues
    df_pval.drop(index=df_pval.index[0], axis=0, inplace=True)
    df_pval['feat_name'] = df_pval.index
    df_pval.reset_index(drop=True, inplace=True)
    df_pval['index'] = df_pval.index
    df_pval = df_pval[['index', 'feat_name', 'pval']]
    df_pval = df_pval.sort_values('pval', ascending=bool)
    
    return df_pval


# Creates a compiled neural network model of any number of layers
# nur_list is a list of number of neurons for each layer
# a_func is the activation function of choice
# opt is the optimazation function of choice
# loss_ is the loss funciton of choice
# num_dims is the dimensionality of data put into the first layer of the neural network
def create_nn(nur_list, a_func, opt, loss_, num_dims):
    
    nn = Sequential()
    nn.add(Dense(nur_list[0], input_dim=num_dims, activation=a_func))

    for i in range(1, len(nur_list)):
        nn.add(Dense(nur_list[i], activation=a_func))

    nn.compile(optimizer=opt, loss=loss_)
    nn.summary()
    
    return nn


################################################################################################
# MAIN SELECTION METHODS
################################################################################################


### CROSS VALIDATION
# Function that preforms cross validation only for the input model and given set of features
# Outputs a summary to show average r2_bar, r2_cv, AIC, and BIC for each fold of the cross validation
def nnCrossValidation(X, y, folds, epo, bs, nur_list, a_func, opt, loss_):
    
        # Splitting data in folds and appropriate sets for training and testing
        ret_list = splitData(X, y, folds)
        
        r2_list = []
        r2_bar_list = []
        aic_val_list = []
        bic_val_list = []
        
        # Iterating through each fold in cross validation. In this case there are generally 5 or 10 folds
        for fold in ret_list:
            
            X_train = fold[0]
            X_test = fold[1]
            y_train = fold[2]
            y_test = fold[3]
            
            # Creating new nn model each iteration for varying number of input dimensions
            model = create_nn(nur_list, a_func, opt, loss_, X_train.shape[1])
            
            # Training and testing model 
            model.fit(X_train, y_train, epochs = epo, batch_size = bs)
            y_pred = model.predict(X_test)
            
            # Calculating r2, r2_bar, AIC, and BIC for each fold and storing
            r2 = r2_score(y_test, y_pred)
            r2_list.append(r2)
            r2_bar_list.append(calc_r2_bar(len(y), X.shape[1], r2))
            aic_val_list.append(calc_aic(y_test, y_pred, X.shape[1]))
            bic_val_list.append(calc_aic(y_test, y_pred, X.shape[1]))
            
        # Averaging values for each fold and storing
        r2_cv_list_final = np.average(r2_list)
        r2_bar_list_final = np.average(r2_bar_list)
        aic_list_final = np.average(aic_val_list)
        bic_list_final = np.average(bic_val_list)
        
        # Printing dataframe to console as formated table 
        print()
        print("FORWARD SELECTION SUMMARY TABLE:")
        print()

        t = pt(['r2_cv', 'r2_bar', 'AIC', 'BIC'])
        t.add_row([r2_cv_list_final, r2_bar_list_final, aic_list_final, bic_list_final])
        print(t)


        
### FORWARD SELECTION
# Function that preforms forward feature selection favoring those feature with the lowest p-values
# Outputs a summary, report table, and graph to show change in r2_bar, r2_cv, AIC, and BIC as input features are changed
def nnForwardSelection(X, y, folds, epo, bs, nur_list, a_func, opt, loss_):

    #Starting with a temporary empty feature dataframe
    x_tmp = pd.DataFrame()
    
    r2_cv_list_final = []
    r2_bar_list_final = []
    aic_list_final = []
    bic_list_final = []
    
    # Creating model to calulate p values
    model = create_nn(nur_list, a_func, opt, loss_, X.shape[1])
    
    #True = sorting ascending order => forward selection
    df_pval = calc_p_values(model, X, y, True, "FORWARD")
    
    add_order_list = df_pval['index'].to_list()
    feature_names_list = df_pval['feat_name'].to_list()
    
    num_feat = 0

    # Iterating through each feature in a list. This order correspondes to the features p-values. Lower is closer to the front of the list
    for each in add_order_list:
        
        num_feat = num_feat + 1
        
        # Adding feature to a temporary dataframe
        x_tmp = pd.concat([x_tmp, X.iloc[:, each]], axis=1)
        
        # Splitting data in folds and appropriate sets for training and testing
        ret_list = splitData(x_tmp, y, folds)
        
        r2_list = []
        r2_bar_list = []
        aic_val_list = []
        bic_val_list = []
        
        # Iterating through each fold in cross validation. In this case there are generally 5 or 10 folds
        for fold in ret_list:
            
            X_train = fold[0]
            X_test = fold[1]
            y_train = fold[2]
            y_test = fold[3]
            
            # Creating new nn model each iteration for varying number of input dimensions
            model = create_nn(nur_list, a_func, opt, loss_, X_train.shape[1])
            
            # Training and testing model 
            model.fit(X_train, y_train, epochs = epo, batch_size = bs)
            y_pred = model.predict(X_test)
            
            # Calculating r2, r2_bar, AIC, and BIC for each fold and storing
            r2 = r2_score(y_test, y_pred)
            r2_list.append(r2)
            r2_bar_list.append(calc_r2_bar(len(y), num_feat, r2))
            aic_val_list.append(calc_aic(y_test, y_pred, num_feat))
            bic_val_list.append(calc_aic(y_test, y_pred, num_feat))
            
        # Averaging values for each fold and storing
        r2_cv_list_final.append(np.average(r2_list))
        r2_bar_list_final.append(np.average(r2_bar_list))
        aic_list_final.append(np.average(aic_val_list))
        bic_list_final.append(np.average(bic_val_list))
        
    feature_list = range(1, X.shape[1] + 1)
    
    # Creating dataframe of values from each feature addition
    df_final = pd.DataFrame()
    df_final['Num_Features'] = feature_list
    df_final['r2_cv'] = r2_cv_list_final
    df_final['r2_bar'] = r2_bar_list_final
    df_final['aic'] = aic_list_final
    df_final['bic'] = bic_list_final
    
    # Printing dataframe to console as formated table 
    print()
    print("FORWARD SELECTION SUMMARY TABLE:")
    print()
    print("Features In Order Added:", feature_names_list)

    t = pt(['Num_Features', 'r2_cv', 'r2_bar', 'AIC', 'BIC'])
    for row in range(0, df_final.shape[0]):
        t.add_row(df_final.iloc[row, :].to_list())
    print(t)
    
    # Generating figures of r2_cv, r2_bar, AIC, and BIC vs number of features
    fig = plt.figure(figsize=(25, 10))
    fig.suptitle('Forward Selection Graphical Summary')    
    
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax1.plot(feature_list, r2_cv_list_final, label="r2_cv", color="green")
    ax1.plot(feature_list, r2_bar_list_final, label="r2_bar", color="red")
    ax1.set_xlabel('Number of Features')
    ax1.legend()
        
    ax2.plot(feature_list, aic_list_final, label="aic", color="blue")
    ax2.plot(feature_list, bic_list_final, label="bic", color="orange")
    ax2.set_xlabel('Number of Features')
    ax2.legend()
    
    fig.show()


### BACKWARD SELECTION
# Function that preforms backward feature selection favoring those feature with the lowest p-values
# Outputs a summary, report table, and graph to show change in r2_bar, r2_cv, AIC, and BIC as input features are changed
def nnBackwardSelection(X, y, folds, epo, bs, nur_list, a_func, opt, loss_):

    x_tmp = pd.DataFrame()

    r2_cv_list_final = []
    r2_bar_list_final = []
    aic_list_final = []
    bic_list_final = []
    
    # Creating model to calulate p values
    model = create_nn(nur_list, a_func, opt, loss_, X.shape[1])

    #False = sorting descending order => backward selection
    df_pval = calc_p_values(model, X, y, False, "BACKWARD")

    rem_order_list = df_pval['index'].to_list()
    feature_names_list = df_pval['feat_name'].to_list()

    num_feat = X.shape[1]
    
    #Starting with a temporary dataframe of all features
    x_tmp = X

    # Iterating through each feature in a list. This order correspondes to the features p-values. Higher is closer to the front of the list
    for each in feature_names_list:
        
        # Splitting data in folds and appropriate sets for training and testing
        ret_list = splitData(x_tmp, y, folds)
        
        r2_list = []
        r2_bar_list = []
        aic_val_list = []
        bic_val_list = []
        
        # Iterating through each fold in cross validation. In this case there are 5 or 10 folds
        for fold in ret_list:
            
            X_train = fold[0]
            X_test = fold[1]
            y_train = fold[2]
            y_test = fold[3]
            
            # Creating new nn model each iteration for varying number of input dimensions
            model = create_nn(nur_list, a_func, opt, loss_, X_train.shape[1])
            
            # Training and testing model 
            model.fit(X_train, y_train, epochs = epo, batch_size = bs)
            y_pred = model.predict(X_test)
            
            # Calculating r2, r2_bar, AIC, and BIC for each fold and storing
            r2 = r2_score(y_test, y_pred)
            r2_list.append(r2)
            r2_bar_list.append(calc_r2_bar(len(y), num_feat, r2))
            aic_val_list.append(calc_aic(y_test, y_pred, num_feat))
            bic_val_list.append(calc_bic(y_test, y_pred, num_feat))
            
        # Averaging values for each fold and storing
        r2_cv_list_final.append(np.average(r2_list))
        r2_bar_list_final.append(np.average(r2_bar_list))
        aic_list_final.append(np.average(aic_val_list))
        bic_list_final.append(np.average(bic_val_list))

        # Droping feature from temporary dataframe
        x_tmp = x_tmp.drop(each, axis=1)
    
        num_feat = num_feat - 1
    
    feature_list = range(1, X.shape[1] + 1)
    feature_list = list(feature_list)
    feature_list = feature_list[::-1]

    # Creating dataframe of values from each feature subtraction
    df_final = pd.DataFrame()
    df_final['Num_Features'] = feature_list
    df_final['r2_cv'] = r2_cv_list_final
    df_final['r2_bar'] = r2_bar_list_final
    df_final['aic'] = aic_list_final
    df_final['bic'] = bic_list_final

    # Printing dataframe to console as formated table
    print()
    print("BACKWARD SELECTION SUMMARY TABLE:")
    print()
    print("Features In Order Removed:", feature_names_list)

    t = pt(['Num_Features', 'r2_cv', 'r2_bar', 'AIC', 'BIC'])
    for row in range(0, df_final.shape[0]):
        t.add_row(df_final.iloc[row, :].to_list())
    print(t)

    # Generating figures of r2_cv, r2_bar, AIC, and BIC vs number of features
    fig = plt.figure(figsize=(25, 10))
    fig.suptitle('Backward Selection Graphical Summary')    

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(feature_list, r2_cv_list_final, label="r2_cv", color="green")
    ax1.plot(feature_list, r2_bar_list_final, label="r2_bar", color="red")
    ax1.set_xlabel('Number of Features')
    ax1.legend()
        
    ax2.plot(feature_list, aic_list_final, label="aic", color="blue")
    ax2.plot(feature_list, bic_list_final, label="bic", color="orange")
    ax2.set_xlabel('Number of Features')
    ax2.legend()

    fig.show()
    
    
### STEPWISE SELECTION
# Function that preforms stepwise feature selection favoring those feature with the lowest p-values but dropping features if they make no improvment(r2_cv) to the model.
# Outputs a summary, report table, and graph to show change in r2_bar, r2_cv, AIC, and BIC as input features are changed
def nnStepwiseSelection(X, y, folds, epo, bs, nur_list, a_func, opt, loss_):

    #Starting with a temporary empty feature dataframe
    x_tmp = pd.DataFrame()
    
    r2_cv_list_final = []
    r2_bar_list_final = []
    aic_list_final = []
    bic_list_final = []
    
    # Creating model to calulate p values
    model = create_nn(nur_list, a_func, opt, loss_, X.shape[1])
    
    #True = sorting ascending order => forward selection
    df_pval = calc_p_values(model, X, y, True, "STEPWISE")
    
    add_order_list = df_pval['index'].to_list()
    feature_names_list = df_pval['feat_name'].to_list()
    
    num_feat = 1
    
    # Setting initial r2_cv value to 0
    r2_cv_old = 0
    
    final_feat_ind_list = []
    final_feat_name_list = []
    final_feat_name_drop_list = []

    # Iterating through each feature in a list. This order correspondes to the features p-values. Lower is closer to the front of the list
    for each in add_order_list:
        
        # Adding feature to a temporary dataframe
        x_tmp = pd.concat([x_tmp, X.iloc[:, each]], axis=1)
        
        # Splitting data in folds and appropriate sets for training and testing
        ret_list = splitData(x_tmp, y, folds)
        
        r2_list = []
        r2_bar_list = []
        aic_val_list = []
        bic_val_list = []
        
        # Iterating through each fold in cross validation. In this case there are 5 or 10 folds
        for fold in ret_list:
            
            X_train = fold[0]
            X_test = fold[1]
            y_train = fold[2]
            y_test = fold[3]
            
            # Creating new nn model each iteration for varying number of input dimensions
            model = create_nn(nur_list, a_func, opt, loss_, X_train.shape[1])
            
            # Training and testing model
            model.fit(X_train, y_train, epochs = epo, batch_size = bs)
            y_pred = model.predict(X_test)
            
            # Calculating r2, r2_bar, AIC, and BIC for each fold and storing
            r2 = r2_score(y_test, y_pred)
            r2_list.append(r2)
            r2_bar_list.append(calc_r2_bar(len(y), num_feat, r2))
            aic_val_list.append(calc_aic(y_test, y_pred, num_feat))
            bic_val_list.append(calc_bic(y_test, y_pred, num_feat))
            
        # Finding r2_cv for feature change
        r2_cv = np.average(r2_list)
        
        # Comparing new r2_cv with old to make sure feature addition offers improvement to model
        if r2_cv > r2_cv_old:
            
            # Update r2_cv_old value to latest r2_cv
            r2_cv_old = r2_cv
            
            num_feat = num_feat + 1
            
            # Store necessary metrics
            final_feat_ind_list.append(each)
            final_feat_name_list.append(feature_names_list[each])
            r2_cv_list_final.append(r2_cv)
            r2_bar_list_final.append(np.average(r2_bar_list))
            aic_list_final.append(np.average(aic_val_list))
            bic_list_final.append(np.average(bic_val_list))
            
        # If there is no model improvement with the addition of a feature the feature is removed. 
        else:
            
            # Feature dropped and recorded
            x_tmp.drop(columns=x_tmp.columns[-1], axis=1, inplace=True)
            final_feat_name_drop_list.append(feature_names_list[each])
        
    feature_list = range(1, num_feat)
    
    # Creating dataframe of values from each feature change
    df_final = pd.DataFrame()
    df_final['Num_Features'] = feature_list
    df_final['r2_cv'] = r2_cv_list_final
    df_final['r2_bar'] = r2_bar_list_final
    df_final['aic'] = aic_list_final
    df_final['bic'] = bic_list_final
    
    # Printing dataframe to console as formated table
    print()
    print("STEPWISE SELECTION SUMMARY TABLE:")
    print()
    print("Features Added:", final_feat_name_list)
    print("Features Dropped:", final_feat_name_drop_list)

    t = pt(['Num_Features', 'r2_cv', 'r2_bar', 'AIC', 'BIC'])
    for row in range(0, df_final.shape[0]):
        t.add_row(df_final.iloc[row, :].to_list())
    print(t)
    
    # Generating figures of r2_cv, r2_bar, AIC, and BIC vs number of features
    fig = plt.figure(figsize=(25, 10))
    fig.suptitle('Stepwise Selection Graphical Summary')
    
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax1.plot(feature_list, r2_cv_list_final, label="r2_cv", color="green")
    ax1.plot(feature_list, r2_bar_list_final, label="r2_bar", color="red")
    ax1.set_xlabel('Number of Features')
    ax1.legend()
        
    ax2.plot(feature_list, aic_list_final, label="aic", color="blue")
    ax2.plot(feature_list, bic_list_final, label="bic", color="orange")
    ax2.set_xlabel('Number of Features')
    ax2.legend()
    
    fig.show()    
    
    
  
    
### More Concise Method For CV Testing
    
# scoring = ['r2', 'neg_mean_squared_error' ]

# import sklearn
# print(sorted(sklearn.metrics.SCORERS.keys()))

# len(df.columns)

# x_tmp = pd.DataFrame()
# results = []

# for each in range(0, X.shape[1]):
    
#     x_tmp = pd.concat([x_tmp, df.iloc[:, each]], axis=1)
    
#     res = cross_validate(model, x_tmp, y, scoring=scoring, cv=10)  # 80-20 split
    
#     results.append(res)