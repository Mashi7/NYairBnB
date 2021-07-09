import pandas as pd
import numpy as np
import NYstats as nys
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score, cross_val_predict, \
    StratifiedShuffleSplit, RepeatedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error

df = nys.df
df.drop(columns = ['host_id', 'geometry', 'last_review', 'price'], inplace = True)
# dropping geometry that was used for map plots
# drop price as well, since using pricelog
# and last review that is just date

def traintestsplit(df, features):
    # Simple train test split (worse results)
    # Y = df["pricelog"]
    # X = df.drop('pricelog', axis = 1)[features]
    # xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.2, random_state = 42)

    # Stratified train test split - accounts for heavy distribution differences between ngbh groups
    # and its influence on the price
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    df = df.reset_index(drop=True)
    for train_index, test_index in split.split(df, df['neighbourhood_group']):
        xtrain = df.loc[train_index, 'neighbourhood_group':'availability_365']
        ytrain = df.loc[train_index, 'pricelog']
        xtest = df.loc[test_index, 'neighbourhood_group':'availability_365']
        ytest = df.loc[test_index, 'pricelog']
    return xtrain, xtest, ytrain, ytest

def preprocess(numerical, categorical):

    # preprocess in pipeline to avoid scaling data leakage in cross validation
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    transformer = ColumnTransformer(transformers=[('num', num_transformer, numerical),
                                                  ('cat', cat_transformer, categorical)])
    return transformer

def cvtrain(models, transformer, xtrain, ytain):

    # split data into kfolds for CV, shuffle used for random distribution of data
    kfolds = 4  # 3/4 used to train, 1/4 fo val, 4 scores for more robust score

    split = KFold(n_splits=kfolds, shuffle=True, random_state=42) # actual Kfold split

    # loop for cross validating models one by one and printing the results
    globmin, secbest, thirdbest = 1, 1.1, 1.2
    minname, secname, thirdname = '', '', ''

    for name, model in models.items():

        model_steps = Pipeline(steps=[
            ('preprocessor', transformer),  # apply previously created pipeline data preprocessor
            ('model', model)],
            verbose=False    # show progress bar
        )
        # create cv results variable for each iteration for each model. neg MAE score -> closer to 0 = better
        # MAE scores are seemingly 'low', because model is CVed on log values of price
        # since this is initial model assessment, the steps to exp the score have not been taken
        cvresults = cross_val_score(model_steps, xtrain, ytrain,
                                    cv=split, scoring='neg_mean_absolute_error',
                                    n_jobs=-1)

        # pull lowest, highest and average R2 score and show std between them
        minscore = -round(min(cvresults), 4)
        maxscore = -round(max(cvresults), 4)
        mean_score = -round(np.mean(cvresults), 4)
        dev = round(np.std(cvresults), 4)

        if mean_score < globmin:
            thirdname = secname
            thirdbest = secbest
            secbest = globmin
            secname = minname
            globmin = mean_score
            minname = name
        elif mean_score > globmin and secbest > mean_score:
            thirdbest = secbest
            thirdname = secname
            secbest = mean_score
            secname = name
        elif mean_score > secbest and thirdbest > mean_score:
            thirdbest = mean_score
            thirdname = name
        print(f"{name} cross val neg MAE score: {mean_score} +/- {dev} (std), min: {minscore}, max:{maxscore}")

    print(f'Best:{minname} with MAE of: {globmin}, Second: {secname} with MAE of: {secbest}, '
              f'Third: {thirdname} with MAE of {thirdbest}')

# performs random grid search for best model hyperparameters
# takes a dict of shortlisted models, list of dicts of hyperparameters space for selected models, train and test sets and
# pipelinee transformer
def randomsearch(bestmodels, hypersettings, xtrain, ytrain, transformer):

    # to make search more robust, it is done with Kfold
    cv = RepeatedKFold(n_splits=3, n_repeats=2, random_state=42)
    # empty lists for best params and best scores
    bestparams = []
    scores = []
    # namelist for indexing and looping purposes
    namelist = list(bestmodels)

    # loop that performs CV and checks for best hyperparams and best scores
    for name, model in bestmodels.items():
        model_steps = Pipeline(steps=[
            ('preprocessor', transformer),  # apply previously created pipeline data preprocessor
            ('model', model)])  # show progress bar

        randomsearch = RandomizedSearchCV(estimator=model_steps, param_distributions=hypersettings[namelist.index(name)],
                                           n_iter=30, cv=cv, verbose=2, random_state=42, n_jobs=-1,
                                           scoring='neg_mean_absolute_error')
        iterbest = randomsearch.fit(xtrain, ytrain)
        bestparams.append(iterbest.best_params_)
        scores.append(iterbest.best_score_)

    # printing best hypereparams and best scores
    for name in namelist:
        print(f'{name} score: {scores[namelist.index(name)]},'
              f'Best Params: {bestparams[namelist.index(name)]}')


def fitmodel(mainmodel, transformer, xtrain, ytrain, numerical, categorical):
    model_steps = Pipeline(steps=[
        ('preprocessor', transformer),  # apply previously created pipeline data preprocessor
        ('model', mainmodel)])


    model_steps.fit(xtrain, ytrain)

    # Recall names of columns that were one hot encoded.
    onehotnames = list(model_steps.named_steps['preprocessor'].
                       named_transformers_['cat'].named_steps['onehot'].
                       get_feature_names(input_features=categorical))


    featurelist = numerical + onehotnames

    weightsimp = np.around(np.mean([tree.feature_importances_ for tree in model_steps.named_steps['model'].estimators_],
                                    axis=0), decimals=4)
    weightwithname = pd.DataFrame(sorted(zip(weightsimp, featurelist), reverse=True), columns=['Weight', 'Feature'])
    print(weightwithname.head(20))
    return model_steps

def modelpredict(model_steps, xtest, ytest):
    ypred = model_steps.predict(xtest)
    predvsactual = pd.DataFrame({'Actual': ytest, 'Predicted': ypred})
    print(predvsactual.head(10))
    print('MAE for the best model:', mean_absolute_error(ytest, ypred))
    print('RMSE for the best model:', np.sqrt(mean_squared_error(ytest, ypred)))

numerical = []
categorical = []

for column in df.columns:
    if df[column].dtype == object:
        categorical.append(column)
    else:
        numerical.append(column)

numerical.remove('pricelog')
features = numerical + categorical

# do 80-20 stratified train test split (distribution of ngbh group)
xtrain, xtest, ytrain, ytest = traintestsplit(df, features)


# # Verify if start split worked
# allxsets = [df, xtrain, xtest]
# for set in allxsets:
#     print(set['neighbourhood_group'].value_counts() / len(set))

# create pipelines to handle numerical and categorical data:
transformer = preprocess(numerical, categorical)    # create pipeline transformer for preprocessing

# # to check how train and test sets look like after pipeline normalization
# xtrain_p = pd.DataFrame(transformer.fit_transform(xtrain).toarray())
# xtest_p = pd.DataFrame(transformer.transform(xtest).toarray())
# print(xtrain_p.head(), '\n',
#       xtest_p.head())

# list of models to train in a form of dictionary
models = {'LIG_model': LinearRegression(),
          'RLR_model': Ridge(),
          'SVR': svm.SVR(kernel='poly', degree=1),
          'lSVR': svm.LinearSVR(),
          'SGDR': SGDRegressor(),
          'kNNR': KNeighborsRegressor(),
          'TreeReg': DecisionTreeRegressor(),
          'MLPreg': MLPRegressor(),
          'rForestReg': RandomForestRegressor(),
          'BagReg': BaggingRegressor(),
          'GradBoostReg': GradientBoostingRegressor(),
          'ExtraTreeReg': ExtraTreeRegressor()}

bestmodels = {'rForestReg': RandomForestRegressor(),
              'BagReg': BaggingRegressor(),
              'GradBoostReg': GradientBoostingRegressor()}

# cvtrain(models, transformer, xtrain, ytrain) # call function that will cross validate models from models dictionary
#shortlist 3 best potential models with cross validation


spacefor = dict()
spacefor['model__n_estimators'] = [1, 10, 50, 100, 500, 1000, 1400, 1800, 2000]
spacefor['model__max_depth'] = [None, 1, 10, 30, 50, 70, 90, 100]
spacefor['model__min_samples_split'] = [2, 5, 7, 10]
spacefor['model__min_samples_leaf'] = [1, 3, 5, 10]
spacefor['model__max_features'] = ['auto', 'sqrt']
spacefor['model__bootstrap'] = [True, False]

spacebag = dict()
spacebag['model__n_estimators'] = [5, 10, 50, 100, 500, 500, 1000, 1500, 2000]
spacebag['model__max_samples'] = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
spacebag['model__bootstrap'] = [True, False]

spacegrad = dict()
spacegrad['model__n_estimators'] = [10, 50, 100, 500, 500, 1000, 2000]
spacegrad['model__learning_rate'] = [0.001, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3]
spacegrad['model__max_depth'] = [1, 2, 3, 5, 7, 9]
spacegrad['model__min_samples_split'] = [2, 5, 7, 10]
spacegrad['model__min_samples_leaf'] = [1, 3, 5, 10]
spacegrad['model__max_features'] = ['auto', 'sqrt']

hypersettings = [spacefor, spacebag, spacegrad]

# randomsearch(bestmodels, hypersettings, xtrain, ytrain, transformer)

# Best initial selection model was bagging regressor with mae of -0.1374
# Run second random search with hyperparameters closer to shortlisted from the first search
spacebag2 = dict()
spacebag2['model__n_estimators'] = [1700, 1900, 2000, 2100, 2300, 2500]
spacebag2['model__max_samples'] = [0.4, 0.45, 0.5, 0.55, 0.6]
spacebag2['model__bootstrap'] = [True, False]

hypersettings2 =[spacebag2]
topmodel = {'BagReg': BaggingRegressor()}

# randomsearch(topmodel, hypersettings2, xtrain, ytrain, transformer)

mainmodel = BaggingRegressor(n_estimators=1700, max_samples=0.5, bootstrap=True, verbose=2)

# Fit the model and obtain weights
model_steps = fitmodel(mainmodel, transformer, xtrain, ytrain, numerical, categorical)

modelpredict(model_steps, xtest, ytest)

# num_transformer[‘scaler’].inverse_transform()