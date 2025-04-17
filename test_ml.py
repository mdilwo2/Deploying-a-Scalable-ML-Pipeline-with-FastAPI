import pytest
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklear.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics
# TODO: add necessary import


# Sample data for testing
data = pd.DataFrame({
    'workclass': ['Private', 'Self-emp-not-inc', 'Private'],
    'education': ['Bachelors', 'HS-grad', 'HS-grad'],
    'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced'],
    'occupation': ['Tech-support', 'Craft-repair', 'Exec-managerial'],
    'relationship': ['Not-in-family', 'Husband', 'Unmarried'],
    'race': ['White', 'Black', 'White'],
    'sex': ['Male', 'Female', 'Female'],
    'native-country': ['United-States', 'United-States', 'United-States'],
    'salary': ['<=50K', '>50K', '<=50K']
})

@pytest.fixture
def processed_data():
    X, y, encoder, lb = process_data(
        data, categorical_features=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'], label='salary', training=True
    )
    return X, y, encoder, lb


# TODO: implement the first test. Change the function name and input as needed
def test_process_data():
    """
    # Test if process_data returns expected result
    """
    
    X, y, encoder, lb = processed_data
    assert isinstance(X, pd.DataFrame), "Expected X to be a DataFrame"
    assert isinstance(y, pd.Series), "Expected y to be a Series"
    assert hasattr(encoder, 'transform'), "Expected encoder to have a transform method"
    assert hasattr(lb, 'transform'), "Expected lb to have a transform method

    pass


# TODO: implement the second test. Change the function name and input as needed
def test_train_model_algorithm(processed_data):
    """
    # Test if the train_model function uses the algorithm
    """
    
    X, y, _, _ = processed_data
    model = train_model(X, y)
    assert isinstance(model, DecisionTreeClassifier), "Expected model to be a DecisionTreeClassifier

    pass


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics_value(processed_data):
    """
    # test if compute_model_metrics_value returns expected results
    """
    
    X, y, _, _ = processed_data
    model = train_model(X, y)
    preds = model.predict(X)
    precision, recall, f1 = compute_model_metrics(y, preds)
    assert precision >= 0 and precision <= 1, "Expected precision to be between 0 and 1"
    assert recall >= 0 and recall <= 1, "Expected recall to be between 0 and 1"
    assert f1 >= 0 and f1 <= 1, "Expected f1 to be between 0 and 

    pass

if __name__ == "__main__":
    pytest.main
