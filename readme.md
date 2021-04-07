# Compare tree based supervised learning algorithms

## About the data
The dataset for this project originates from the UCI Machine Learning Repository. The datset was donated by Ron Kohavi and Barry Becker, after being published in the article "Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid". The article by Ron Kohavi can be found here online. The data we investigate here consists of small changes to the original dataset, such as removing the 'fnlwgt' feature and records with missing or ill-formatted entries.

## Targets of Investigation
Target is to demonstrate how to use tree based machine learning models including all steps tpo be taken for preparing the data.

In addition to that we compare different machine learning models.

## Methodology
1. Data Preparation
   * Transforming Skewed Continuous Features by using logarithmic transformation 
    A dataset may sometimes contain at least one feature whose values tend to lie near a single number, but will also have a non-trivial number of vastly larger or smaller values than that single number. Algorithms can be sensitive to such distributions of values and can underperform if the range is not properly normalized.  It is common practice to apply a logarithmic transformation on the data so that the very large and very small values do not negatively affect the performance of a learning algorithm. Using a logarithmic transformation significantly reduces the range of values caused by outliers. Care must be taken when applying this transformation however: The logarithm of 0 is undefined, so we must translate the values by a small amount above 0 to apply the the logarithm successfully.
    * Normalizing Numerical Features
    In addition to performing transformations on features that are highly skewed, it is often good practice to perform some type of scaling on numerical features. Applying a scaling to the data does not change the shape of each feature's distribution. However, normalization ensures that each feature is treated equally when applying supervised learners. Note that once scaling is applied, observing the data in its raw form will no longer have the same original meaning, as exampled below.'''
    * Transform Non-numeric Values into Numerics
    In our data there are several features for each record that are non-numeric. We need to convert these non-numeric features (called categorical variables) into numeric values. Categorical variables we arey using the one-hot encoding scheme. One-hot encoding creates a "dummy" variable for each possible category of each non-numeric feature.
    * Shuffle and Split Data
    Now all categorical variables have been converted into numerical features, and all numerical features have been normalized. For training the models we will now split the data (both features and their labels) into training and test sets. 80% of the data will be used for training and 20% for testing.
2. Metrics for Model Evaluation
    *  F-score
    F-score is a measure of a test's accuracy. It is calculated from the precision and recall of the test, where the precision is the number of true positive results divided by the number of all positive results, including those not identified correctly, and the recall is the number of true positive results divided by the number of all samples that should have been identified as positive. The F score is a metric which takes in mind, that depending on the underlying issue precision matters more than recall or vice versa.
3. Comparison of Learning Models
   * Decision criteria: <br>
      * Separability: How clustered are the data.
      * Distribution: The target group (data which has to be identified as positive) is very small compared to the total population.
      * Independence: Do the independent features have correlation among them.
      * Dataset size: How many data points are in the population.
      * Features size: How many independent variables are there.
      * Useability: How easy we can parametrize the model.
      * Overfitting: Can the model transfer learning results from training to testing data.
   * Decision matrix for certain learning models:
  
    | Method | Separability | Distribution | Independence | Dataset | Features | Usability | Overfitting |
    | ------ | ------------ | - | - |------- | -------- | --------- | ----------- |
    | **Logistic Regression** | **linear** | balanced | dependent | small | few | easy | fragile |
    | **Gaussian Naive Bayes** | non-linear | balanced | **independent** | small | **many** | easy | fragile |
    | **Decision Trees** | non-linear | balanced | dependent | small | few | easy | fragile |
    | **Random Forrest** (Ensemble Method) | non-linear | balanced | dependent | small | few | easy | **robust** |
    | **Bagging** (Ensemble Method) | non-linear | balanced |  dependent | small | few | easy | fragile |
    | **AdaBoost** (Ensemble Method) | non-linear | **unbalanaced**|  dependent | **large** | few | easy | fragile |
    | **Support Vector Method** | non-linear | balanced | dependent | small | **many** | easy | fragile |
    | **K-Nearest Neighbors** (KNeighbors) | non-linear | balanced | dependent | small | few | easy | fragile |

4. Train and Predict
   * Run the train and predict methods of sklearn for the nominated learning models.
5. Choose the Best Model
    * Based on accuracy and F-score (with beta=0.5 to emphasize the precision) the best model is selected (see results)
6. Model Tuning
   For finding the best parameters we can use systamic search algorithms provided by sklearn.
    * GridSearchCV
    * RandomizedSearchCV
  
   As a metric for comparision we take the F-score with beta=0.5
7. Feature Importance
   Extract the 5 most important features.
8. Feature Selection
    Reduce the data set to the extracted features and retrain the model.

## Results
1. Choose the Best Model
    For the underlying project we got this values for the decision criteria:
    | Separability | Distribution | Independence | Dataset | Features |
    |-|-|-|-|-|
    | non-linear | slightly unbalanced (25/75) |  dependent | small (45222 records) | small (103 total features) |

    Based on this we selected these models:
    | Method | Separability | Distribution | Independence | Dataset | Features |
    |-|-|-|-|-|-|
    |Random Forrest| non-linear | balanced |  dependent | small | few | 
    |AdaBoost|non-linear | unbalanced |  dependent | large | few | 
    |Support Vector Method|non-linear | balanced |  dependent | small | many | 

    * Based on accuracy and F-score (with beta=0.5 to emphasize the precision) with selected the AdaBoost Model as the best learning algorith.
2. Extract the most important feature and retrain the model with this subset
    We see that the top five most important features contribute more than half of the importance of all features present in the data. 


    After retraining the model with the extracted features we lost around 2% of accuracy and around 3% of F-score. We gained 21% in prediction time. For this size of dataset it is not worth to loose quality in order to gain performance but for larger datasets we should consider taking the reduced model.

    |     Metric     | Naive Predictor  | Unoptimized Model | Optimized Model | Reduced Model   |
    | :------------: | :--------------: | :---------------: | :-------------: | :-------------: |
    | Accuracy Score | 0.2478           | 0.8576            | 0.8652          | 0.8417          |
    | F-score        | 0.2917           | 0.7246            | 0.7401          | 0.7017          |
    | Prediction Time| -                | -                 | 0.58            | 0.46            |

# Sources

| Method | Source |
| :----- | :----- |
| **Logistic Regression** | https://towardsdatascience.com/quick-and-easy-explanation-of-logistics-regression-709df5cc3f1e |
| **Gaussian Naive Bayes** | https://towardsdatascience.com/all-about-naive-bayes-8e13cef044cf |
| **Decision Trees** | https://towardsdatascience.com/decision-tree-ba64f977f7c3 |
| **Random Forrest** (Ensemble Method) | https://builtin.com/data-science/random-forest-algorithm<br>https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76 |
| **Bagging** (Ensemble Method) |  |
| **AdaBoost** (Ensemble Method) | https://hackernoon.com/under-the-hood-of-adaboost-8eb499d78eab<br>https://corporatefinanceinstitute.com/resources/knowledge/other/boosting/ |
| **Support Vector Method** |  |
| **K-Nearest Neighbors** (KNeighbors) | https://medium.com/@alex.ortner.1982/top-10-binary-classification-algorithms-a-beginners-guide-feeacbd7a3e2<br>https://towardsdatascience.com/tagged/real-world-examples-knn |
| General | https://towardsdatascience.com/part-i-choosing-a-machine-learning-model-9821eecdc4ce |