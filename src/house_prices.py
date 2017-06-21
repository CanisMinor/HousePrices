import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import math
import sklearn.ensemble as sk
plt.style.use('ggplot')

#region Model performance evaluation functions.
def root_mean_squared_error(predictions, labels):
    '''
    function: root_mean_squared_error
    --------------------------------
    This function returns the root mean squared error (RMSE) for a given set of predicted and actual house values,

    RMSE = sqrt(sum_i(y_i - Y_i)^2 / n),

    where y_i are the predicted values, Y_i are the actual values, n is the number of samples, and sqrt() denotes the
    square root function.

    The lower the RMSE, the closer the fit of the model to the data.

    parameters
      predictions: list of predicted house values (float array)
      labels: list of corresponding actual house values (float array)
    '''

    n_data_points = len(predictions)
    RMSE = 0.0

    for i in range(0, n_data_points):
        RMSE += (predictions[i] - labels[i])**2

    RMSE = math.sqrt(float(RMSE)/float(n_data_points))

    return RMSE

def coefficient_of_determination(predictions, labels):
    '''
    function: coefficient_of_determination
    --------------------------------------
    This function returns the coefficient of determination (R_squared) for a given set of predicted and actual house
    values,

    R_squared = sum_i(y_i - avg({y_i}))^2 / sum_i(Y_i - avg({Y_i}))^2 ,

    where y_i are the predicted values, Y_i are the actual values, and avg({x}) denotes the average of values {x_0, x_1,
     ..., x_n}.

    The coefficient of determination indicates how much of the variances is explained by the model.

    The coefficient of determination has a range of [0,1]; the closer to 1, the better the model

    parameters
      predictions: list of predicted house values (float array)
      labels: list of corresponding actual house values (float array)
    '''

    #find number of datapoints
    n_data_points = len(predictions)

    #calculate the average label value
    avg_label = np.mean(labels)

    #initialise sums
    total_sum_squared = 0.0
    sum_squared_residuals = 0.0

    #evaluate total squared sum of variances, and the total sum of squared residuals
    for i in range(0, n_data_points):
        total_sum_squared += (labels[i] - avg_label)**2
        sum_squared_residuals += (labels[i] - predictions[i])**2

    #evaluate the coefficient of determination, R_squared
    R_squared = 1.0 - (float(sum_squared_residuals)/float(total_sum_squared))

    return R_squared

def standardised_residuals(predictions, labels, n_features, name):
    '''
    function: standardised_residuals
    --------------------------------
    plots the standardised residuals of predicted and actual house prices; if we have made a good choice of model, the
    residuals should be randomly scattered with respect to the values predicted

    parameters:
      predictions: list of predicted house values (float array)
      labels: list of corresponding actual house values (float array)
      n_features: number of features (int)
      name: name of dataset for plot title (string)

    returns: boolean (success/failure)
    '''

    #find nunmber of datapoints in test set
    n_data_points = len(predictions)

    #initialise normalisation factor
    denom = n_data_points - n_features - 1

    #evaluate raw residuals
    residuals = labels - predictions  #elementwise subtraction
    sum_squared_residuals = np.sum(residuals**2)

    #evaluate normalisation factor
    norm_fac = math.sqrt(float(sum_squared_residuals)/float(denom))

    #rescale the residuals to standardised residuals
    residuals /= norm_fac

    #plot residuals versus house prices
    plt.scatter(predictions, residuals, s=20, c=None, marker='x')
    plt.title('residuals plot for ' + name + ' data')
    plt.ylabel('standardised residual')
    plt.xlabel('property price (prediction)')
    plt.show()

    #return success
    return True
#endregion

#region Data pre-processing helpers.
def strp_time_parser(x):
    '''
    function strp_time_parser
    -------------------------
    parses date into datetime format

    parameters:
        x: input to be parsed to datetime format

    returns: the parsed date, or (in case of ValueError), the original input
    '''

    try:
        return dt.datetime.strptime(x, '%Y-%m-%d %H:%M')
    except ValueError:
        return x

def strip(text):
    '''
    function strip
    -------------------------
    strips whitespace and newlines from text

    parameters:
        x: text to be stripped of whitespace and newlines

    returns: the stripped text, or (in case of AttributeError), the original input
    '''

    try:
        return text.strip()
    except AttributeError:
        return text
#endregion

def train_and_test_model(fileName):
    '''
    function: train_and_test_model
    ------------------------------
    reads in and cleans the data, splits the data into training and test sets, trains a random forest regressor on the
    training data and evaluates the performance of the trained random forest regressor

    parameters:
        fileName: path to data file plus filename

    returns: boolean (success/failure)
    '''

    #read in data, stripping whitespace entries, converting dates to datetime format and naming the relevant columns
    print('reading data')
    col_names = ['', 'price', 'date', '', 'property_type', '', 'duration','','','','','town','','','','']
    house_prices_df = pd.read_csv(fileName, names=col_names, parse_dates=[2], date_parser=strp_time_parser, header=None,
                      converters={'town' : strip, 'property_type' : strip, 'duration' : strip, 'date' : strip})

    print('cleaning and formatting data:')
    #drop the columns we don't need to save on memory
    print('  1) dropping irrelevant columns')
    house_prices_df.drop(house_prices_df.columns[[0, 3, 5, 7, 8, 9, 10, 12, 13, 14, 15]], axis=1, inplace=True)

    #convert town to 1 for LONDON and to 0 for all other towns; rename 'town' column to more sensible name 'in_london'
    print('  2) encoding location to inside/outside London')
    house_prices_df.loc[house_prices_df['town'] != 'LONDON', 'town'] = 0
    house_prices_df.loc[house_prices_df['town'] == 'LONDON', 'town'] = 1
    house_prices_df.rename(columns={'town': 'in_london'}, inplace=True)

    #do one-hot encoding of the remaining categorical features (i.e., duration and property_type), since these
    #categories are non-ordinal; do the one-hot encoding in-place with the concatenation so as to save on memory
    print('  3) performing one-hot encoding of duration and property_type features')
    house_prices_df = pd.concat([house_prices_df,pd.get_dummies(house_prices_df[['property_type', 'duration']])],axis=1)
    house_prices_df.drop(house_prices_df.columns[[2,3]], inplace=True, axis=1)

    #create test data set by subsetting the original data frame (picking all entries from 2015)
    print('creating test dataset')
    test_df = house_prices_df[house_prices_df['date'].dt.year == 2015]

    #create training data set by deleting the test data set from the original house_prices_df data frame
    #note house_prices_df is now the training data set
    print('creating training dataset')
    house_prices_df = house_prices_df[house_prices_df['date'].dt.year != 2015]

    #train a random forest on the training data:
    # use 100 decision trees (n_estimators=100)
    # give every decision tree access to all features (max_features = 1)
    # require at least 10 000 samples for a node to split (min_samples_split=10000) for regularisation purposes
    # parallelise training to all available cores (n_jobs = -1)
    print('training random forest regressor')
    regressor = sk.RandomForestRegressor(n_estimators=100, min_samples_split=10000, n_jobs=-1, max_features=1)
    regressor.fit(house_prices_df.iloc[:,2:], house_prices_df.iloc[:,0])

    #evaluate predictions of the trained random forest on the test dataset
    print('evaluating predictions on test and training set')
    test_results = regressor.predict(test_df.iloc[:,2:])
    train_results = regressor.predict(house_prices_df.iloc[:,2:])

    #evaluate the root mean squared error and compare to scale (range) of prices in dataset
    print('evaluating performance of model on test data:')
    std_dev_price_test = np.std(test_df.iloc[:,0])
    avg_price_test = np.mean(test_df.iloc[:,0])
    RMSE_test = root_mean_squared_error(test_results, test_df.iloc[:,0].as_matrix())
    print('  1) root-mean-squared error on test set: ' + str(RMSE_test))
    print('     The RMSE should be compared to the general range of prices in the data:')
    print('        test set: average price ' + str(avg_price_test) + ', standard dev. ' + str(std_dev_price_test))

    #evaluate the coefficient of determination, R_squared
    R_squared_test = coefficient_of_determination(test_results, test_df.iloc[:,0].as_matrix())
    R_squared_train = coefficient_of_determination(train_results, house_prices_df.iloc[:,0].as_matrix())
    print('  2) coefficient of determination for test data: ' + str(R_squared_test))
    print('     versus coefficient of determination for training data: ' + str(R_squared_train))

    #evaluate quality of results, by plotting residuals as a function of predicted prices
    n_features = len(house_prices_df.columns) - 2   #subtract two for the 'price' and 'date' columns
    print('  3) plotting residuals plot')
    standardised_residuals(test_results, test_df.iloc[:,0].as_matrix(), n_features, 'test')

    return True

train_and_test_model('../data/pp-complete.csv')