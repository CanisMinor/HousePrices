__author__ = 'cjaas'


'''
#in London
    subsetA = house_prices_df[house_prices_df['in_london'] == True]
    for property_type in ['D', 'S', 'T', 'F', 'O']:
        subsetB = subsetA[subsetA['property_type'] == property_type]
        for duration in ['L', 'F']:
             subsetC = subsetB[subsetB['duration'] == duration]
             if len(subsetC) > 0:
                alpha, beta = np.polyfit(mdates.date2num(subsetC['date']), subsetC['price'], 1) #try a linear model
                trained_model_parameters.update({(True, property_type, duration) : (alpha, beta)})

    #outside London
    subsetA = house_prices_df[house_prices_df['in_london'] == False]
    for property_type in ['D', 'S', 'T', 'F', 'O']:
        subsetB = subsetA[subsetA['property_type'] == property_type]
        for duration in ['L', 'F']:
             subsetC = subsetB[subsetB['duration'] == duration]
             if len(subsetC) > 0:
                alpha, beta = np.polyfit(mdates.date2num(subsetC['date']), subsetC['price'], 1) #try a linear model
                trained_model_parameters.update({(False, property_type, duration) : (alpha, beta)})


#region Data exploration.
def plot_house_prices():
   function: plot_house_prices
   ---------------------------
   This function plots the house prices for a given type of property for a given value of in/out London, as a function of the lease length.

   parameters:
      house_prices: the house prices (float)
      property_type: the property type (numerical category)
      london_inout: is the property in London or not (bool)

   return True

#endregion

'''
def relative_squared_error(predictions, labels):
    '''
    function: relative_squared_error
    --------------------------------
    This function returns the relative squared error (RSE) for a given set of predicted and actual house values,

    RSE = sum_i(y_i - Y_i)^2 / sum_i(avg({Y_i}) - Y_i)^2 ,

    where y_i are the predicted values, Y_i are the actual values, and avg({x}) denotes the average of values {x_0, x_1, ..., x_n}.

    The RSE thus lies in the range [0,1].  The closer to 1, the better the fit.

    parameters
      predictions: list of predicted house values (float array)
      labels: list of corresponding actual house values (float array)
    '''
#test_df = house_prices_df[house_prices_df['date'].apply(lambda x : x.year == 2015)]

    #std_dev_price_train = np.std(house_prices_df.iloc[:,0])
    #avg_price_train = np.mean(house_prices_df.iloc[:,0])

    #print('      - training set: average price ' + str(avg_price_train) + ', standard dev. ' + str(std_dev_price_train))
#print('     versus coefficient of determination for training data: ' + str(R_squared_train))


#standardised_residuals(train_results, house_prices_df.iloc[:,0].as_matrix(), n_features, 'training')

RMSE_train = root_mean_squared_error(train_results, house_prices_df.iloc[:,0].as_matrix())



    nom = 0.0
    denom = 0.0
    n_data_points = len(predictions)
    avg_label = np.mean(labels)

    for i in range(0, n_data_points):
        nom += (predictions[i] - labels[i])**2
        denom += (avg_label - labels[i])**2

    return float(nom/denom)