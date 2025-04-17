# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model Name: Census Income Prediction Model
Model Type: Supervised Machine Learning
Model Architecture: Decision Tree Classifier
Input Data: Census Data
Output: Predicted income category

## Intended Use
The predicive model is designed to predict whether an individuals income is less than 50K or over 50K based on demographic related features.

## Training Data
The model was trained on the census.csv dataset. This dataset includes demographic and employment information. The dataset was split into 80% training and 20% test sets to evaluate the performance.

## Evaluation Data
The data was used to assess the models accuracy, precision, recall, and f1 score.

## Metrics
Precision: Measures accuracy of positive predictions.
Recall: Measures ability of the model to identify relevent instances.
F1 Score: Provides a single metric for model performance.

## Ethical Considerations
It is important to consider ethical implications. This could potentially be bias in the training data, and the potential impact on individuals. There should be efforts taken to ensure the model does not disproportionatley affect any particular group.
## Caveats and Recommendations
The model generally performs well on the dataset but it may have limitations when applied to specific subgroups or categories. For instance the models performance varies across different workclasses. 
