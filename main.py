import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# the train_test_split_function allows for splitting up
# data to reserve some for testing
from sklearn.model_selection import train_test_split 
# LinearRegression model from the linear_model module
from sklearn.linear_model import LinearRegression

def main():
    # write your code here
    #retrieve the data/records
    pd.set_option('display.width', None)
    file_path = 'diabetes.csv'
    df_diabetes = pd.read_csv(file_path, skiprows = 1)

    #select attriutes
    #identify target attribute (columns)
    #select the attribute most statistically correlated with the target
    #deal with duplicates values
    df_diabetes = df_diabetes.drop_duplicates()
    df_diabetes = df_diabetes[['BMI', 'Y']]

    #deal with the missing values
    df_diabetes = df_diabetes.dropna()
    print(df_diabetes.info())
    
    
    #separate the data into target (output) and feature vector(input)
    x = df_diabetes['BMI']
    y = df_diabetes['Y']

    # create the x-value into a 2D array from a 1D series 
    X = pd.DataFrame(x.values.reshape(-1,1), columns = [x.name])
   
    #train the model
    model_linreg = LinearRegression()
    model_linreg.fit(X, y)

    #predict the model and create the line of best fit
    X_trend = np.array([[x.min()], [x.max()]])
    y_pred = model_linreg.intercept_ + model_linreg.coef_[0]*X_trend
    y_pred = model_linreg.predict(X_trend)
   
    
    #scatter plot of target attribute vs most statistically-correlated attribute
    fig,ax = plt.subplots(1,1, figsize = (16,8))
    ax.scatter(X,y, label = 'Diabetes data')

    #line plot of line of best fit 
    ax.plot(X_trend, y_pred, color= 'orange', label = 'Line of best fit')

    #x-label, y-label, title, and legend
    
    binwidth = 2.5
    plt.xlabel('BMI')
    plt.ylabel('Progression')
    ax.legend()
    fig.suptitle('Diabetes Data: Progression vs. BMI (Linear Regression)')
    fig.tight_layout()

    #save the figure
    plt.savefig('Diabetes_Regresssion.png')



if __name__ == '__main__':
    main()
