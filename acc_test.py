########################################################################################################################
###########################################  Accuracy Testing Script  ##################################################
########################################################################################################################

# Import libraries

import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import libraries
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

def test(type:str) -> tuple:

    # Download the dataset
    url = "Egg_Production.csv"
    eggs = pd.read_csv(url, sep=',')

    # Let's create a backup copy of the dataset
    egg_backup = eggs.copy()

    # Removing duplicates
    eggs.drop_duplicates(inplace=True)

    #
    # Not turning to list doesn't seem to change functionality for line 6
    #

    # Store columns into list
    num_cols = eggs.select_dtypes(include='number').columns

    # Select MinMaxScaler
    SCALER = MinMaxScaler()

    # Make pipeline
    num_pipeline = make_pipeline('num', SCALER, num_cols)

    # Create preprocessing
    preprocessing = ColumnTransformer([num_pipeline], remainder='passthrough')

    # Fit and transform dataframe
    eggs_scaled = preprocessing.fit_transform(eggs)

    # Prepare new data set
    feature_names = preprocessing.get_feature_names_out()
    eggs_prepared = pd.DataFrame(data=eggs_scaled, columns=feature_names)

    VALIDATION_SIZE =  0.20
    TEST_SIZE = 0
    TRAIN_SIZE = 1 - VALIDATION_SIZE - TEST_SIZE
    SEED = None # Make None type to Randomize

    from sklearn.model_selection import train_test_split

    X = eggs_prepared.drop(["num__Total_egg_production"], axis=1) # it's not a feature but a target
    y = eggs_prepared["num__Total_egg_production"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                        test_size=VALIDATION_SIZE, 
                                        train_size=TRAIN_SIZE, 
                                        random_state=SEED, 
                                        shuffle=True)

    # Define utility functions for later!

    ################################### Rangeplot ###################################
    # Takes 2 np.ndarray's of numbers and makes a rangeplot with their differences
    # Takes optional title

    def make_rangeplot(predictions: np.ndarray, actual_values=y_test, title='', x_labels=None,axs=None):
        """ Makes a rangeplot """
        bottoms = np.minimum(predictions, actual_values)
        tops = np.maximum(predictions, actual_values)

        diff = np.subtract(tops, bottoms)

        x = np.arange(len(bottoms))
        
        axs.bar(x, diff, bottom=bottoms)
        title += ' Rangeplot'
        axs.set_ylabel('Prediction Value')
        axs.set_xlabel('Prediction Number')
        axs.set_xticks(x, x_labels)
        axs.set_title(title)

    ############################# Scatterplot #######################################
    # Takes 2 np.ndarray's of numbers and makes a scatterplot comparing predictions to actual values
        
    def make_scatterplot(pred_values: np.ndarray, act_values=y_test, title=''):
        plt.scatter(pred_values, act_values)  # y is your actual target values
        plt.xlabel("Predicted Values")
        plt.ylabel("Actual Values")
        title += " Predicted vs. Actual Values"
        plt.title(title)
        plt.axline((0, 0), slope=1)
        plt.show()

    ############################## Print Stats ######################################
    # Takes predictions and actual values, and prints MSE, MAE, and R^2 values

    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    def print_stats(pred_values: np.ndarray, act_values=y_test):
        print(f'Mean Squared Error:\t\t{mean_squared_error(act_values, pred_values):.5f}')           # Lower better
        print(f'Mean Absolute Error:\t\t{mean_absolute_error(act_values, pred_values):.5f}\n')       # Lower better
        print(f'Coefficient of Determination:   {r2_score(act_values, pred_values):.5f}')            # Higher better

    model = 0

    if(type == 'lin'):
        # Run linear regression on our model
        from sklearn.linear_model import LinearRegression

        lin_reg_model = LinearRegression()

        lin_reg_model.fit(X_train, y_train)

        model = lin_reg_model.predict(X_test)

    elif(type == 'grad'):
        # Run GBR on our data
        from sklearn.ensemble import GradientBoostingRegressor

        grad_boosting_model = GradientBoostingRegressor()

        grad_boosting_model.fit(X_train, y_train)

        model = grad_boosting_model.predict(X_test)

        
    else:
        from sklearn.ensemble import RandomForestRegressor

        # Run forest regression model on our data
        rand_forest_model = RandomForestRegressor()

        rand_forest_model.fit(X_train, y_train)

        model = rand_forest_model.predict(X_test)

    return (mean_squared_error(y_test, model), mean_absolute_error(y_test, model), r2_score(y_test, model))

results = list()

for i in range(1000):
    results.append(test(''))

mse = 0
mae = 0
r2  = 0

for i in range(len(results)):
    mse += results[i][0]
    mae += results[i][1]
    r2  += results[i][2]

print(f'MSE: {mse/len(results)}')
print(f'MAE: {mae/len(results)}')
print(f'R^2: {r2/len(results)}')