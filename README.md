# NYairBnB
Small project of data analysis and machine learning of New York AirBnB Data (kaggle.com/dgomonov/new-york-city-airbnb-open-data) in Python.

## NYstats 
contains statistical analysis with geo plotting that was later used to determine feature selection in ML section. Analysis is comprised of:

* function for obtaining statistical info on raw data and on data grouped by criteria

* plotting airbnb offers by price on NY map

* function for plotting price histogram without outliers for data divided into custom groups

* plotting airbnb offers on a map with district division and price scale

* function for pie chart plotting to show % distribution by custom groups

* bar chart to show median and mean listing price per district and differences between them

* correlation heatmap for features

* scatterplot showing correlation of number of reviews with price for listings divided into room_type 

* matrix of scatterplots for selected attributes to show their correlation in pairs

## NYML

contains ML project with regression task to predict listing prices of AirBnB in New York. The project goes through several stages:

1.  Feature selection (rejection of features that proved not to be important in NYstats
2.  Feature engineering:

    * division into numerical and categorical
    * creating separate pipelines for numrical and categorical features (including: filling missing info, scaling, onehot encoding)
    * joining the transformers into one Pipeline
3.  Stratified Train-Test split (to acknowledge different price distribution among neighbourhood_group)
4.  Passing a dictionary of potential regression models to simple Cross Validation on Train Set to shortlist candidates for final model selection
5.  Random searching Cross Validation in a space of hyperparameters for 3 most promising models
6.  Doing a secondary random search with narrower hyperparameters range for the best model
7.  Fitting the best model and displaying weight importances for top 20 features
8.  Using test set to predict values for price, compare with actual values and display metrics 
