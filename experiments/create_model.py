import json
import pathlib
import pickle
from typing import List
from typing import Tuple

import pandas
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from catboost import Pool, CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/kc_house_data.csv"  # path to CSV with demographics
# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode', 'waterfront', 'view', 'condition', 
    'grade', 'yr_built', 'yr_renovated'
]

# Define categorical features
CATEGORICAL_FEATURES = ['waterfront', 'view', 'condition', 'grade', 'yr_built', 'yr_renovated']

OUTPUT_DIR = "model"  # Directory where output artifacts will be saved


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with home sale data
        sales_column_selection: list of columns from sales data to be used as
            features

    Returns:
        Tuple containg with two elements: a DataFrame and a Series of the same
        length.  The DataFrame contains features for machine learning, the
        series contains the target variable (home sale price).

    """
    data = pandas.read_csv(sales_path,
                           usecols=sales_column_selection,
                           dtype={'zipcode': str})
    demographics = pandas.read_csv("data/zipcode_demographics.csv",
                                   dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left",
                             on="zipcode").drop(columns="zipcode")
    # Remove the target variable from the dataframe, features will remain
    y = merged_data.pop('price')
    x = merged_data

    return x, y

def validate_model(model, features, y_train, x_test, y_test):
    y_test_pred = model.predict(x_test[features])
    y_test_naive = [y_train.mean()]*len(y_test)
    

    mae_naive = mean_absolute_error(y_test, y_test_naive) 
    r2_naive = r2_score(y_test, y_test_naive) 
    mae_test =  mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    return mae_naive, r2_naive, mae_test, r2_test

def main():
    """Load data, train model, and export artifacts."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    x_train, _x_test, y_train, _y_test = model_selection.train_test_split(
        x, y, random_state=42)



    # Define the grid of hyperparameters to search
    params = {
        'learning_rate': [0.03, 0.1, 0.3],
        'depth': [4, 6, 8],
        'iterations': [50, 100, 200]
    }
    
    # initialize the model
    cat = CatBoostRegressor(random_seed=42, silent=False, cat_features=[5,6,7,8,11,12])
    
    # perform grid search with 5-fold cross-validation
    grid_search = GridSearchCV(cat, param_grid=params, cv=5, scoring='neg_mean_absolute_error')
    
    # fit the grid search to the data
    grid_search.fit(x_train,y_train)
    
    # Get the best model based on the chosen evaluation metric
    model = grid_search.best_estimator_

    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Output model artifacts: pickled model and JSON list of features
    pickle.dump(model, open(output_dir / "model.pkl", 'wb'))
    json.dump(list(x_train.columns),
              open(output_dir / "model_features.json", 'w'))

    # Generate validation metrics for the run
    mae_naive, r2_naive, mae_test, r2_test = validate_model(model, x_train.columns, y_train, _x_test, _y_test)
    
    with open("metrics.txt", "w") as outfile:
        outfile.write("Mean Absolute Error naive (mean of the prices on training dataset): " + str(mae_naive) + "\n")
        outfile.write("R-squared Score naive (mean of the prices on training dataset): " + str(r2_naive) + "\n")
        outfile.write("Mean Absolute Error (test dataset): " + str(mae_test) + "\n")
        outfile.write("R-squared Score (test dataset): " + str(r2_test) + "\n")


if __name__ == "__main__":
    main()
