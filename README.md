# student
This repository is created for AI SG AI AP two days Bootcamp AI project

README.md 
a. Full name: NATHAN ONG KEE WEE
    Email address: KERHL8@GAMIL.COM
b. Overview of the submitted folder and the folder structure
	This repository is to have the following structure:
	|------ .github
	|------ src
	|	Python files constituting end-to-end ML pipeline in .py format
	|------ README.md
	|------ eda.ipynb
	|------ requirements.text
	|____ run.sh
c. Instructions for executing the pipeline and modifying any parameters
parser.usage =  "Type the command: python student.eda.py -v(erbose) <#>" \
                "# - 1 for 'student.head'" \
                "# - 2 for 'student.describe'" \
                "# - 3 for 'student.dtypes'" \
                "# - 4 for 'student.columns'" \
                "# - 5 for 'student.hist'" \
                "# - 6 for 'sns.boxplot'" \
                "# - 7 for 'sns.pairplot(student)'" \
                "# - 8 for 'student.shape'" \
                "# - 9 for 'student.count'" \
                "# - 10 for 'duplicated_student.shape'" \

parser.usage =  "Type the command: python student.prepare_data.py -v(erbose) <#>" \
                "# - 1 for 'student.drop_duplicates()'" \
                "# - 2 for 'student.drop() for redundant columns of index, student_id, sleep_time, and wake_time'" \
                "# - 3 for 'student.isnull().sum() for data containing null values'" \
                "# - 4 for 'student.isnull().sum() for final_test and attendance_rate'" \
                "# - 5 for 'For calculating the skewness of the numerical data'" \
                "# - 6 for 'For split the data into features and target labels'" \
                "# - 7 for 'For Log-transform the skewed features, check'" \
                "# - 8 for 'For features_log_transformed'" \
                "# - 9 for 'For For train_test_split data and One-hot encode the  'features_log_minmax_transform' data using pandas.get_dummies() and For calculating Naives Bayes using TP, FP, TN, FN'" \

parser.usage =  "Type the command: python student.prepare_data.py -v(erbose) <#>" \
                "# - 1 for 'Pipeline 1: Get predictions on the first 300 training samples(X_train)'" \
                "# - 2 for 'Pipeline 2: Results, accuracy, fbeta_score on three models of SVC,DecisionTreeClassifier,RandomForestClassifier'" \
                "# - 3 for 'Printing out the values for 1%, 10% and 100%'" \
                "# - 4 for 'Use gridsearch to fit and predict accuracy and fbeta scores'" \
                "# - 5 for 'Fit, train and obtain top 5 feature importances features on the chosen model'" \
                "# - 6 for 'Obtain accuracy and fbeta scores on final model'" \


d. Description of logical steps/flow of the pipeline
The student data is broken down into three main python files, namely: student_eda.py, student_prepare_data.py and student_MLPiple.py. The file student_eda.py primarily contains the blocks of codes to be executed in order to make sense of the data we have. The file student_prepare_data.py will involve cleaning, dropping redundant and null data, transforming and normalising skewed data as well as Hot encoding discrete datasets. The last file will be involved in the actual Machine Learning processes.
In the student_eda.py file, 10 data exploration steps are performed to grasp the presented dataset. These 10 steps are:
1. for 'student.head'
2. for 'student.describe'
3. for 'student.dtypes'
4. for 'student.columns'
5. for 'student.hist'
6. for 'sns.boxplot'
7. for 'sns.pairplot(student)'
8. for 'student.shape'
9. for 'student.count'
10. for 'duplicated_student.shape'

In the student_prepare_data.py file, the actual process of preparing the data for use is contain in this file.

1. for 'student.drop_duplicates()'
2. for 'student.drop() for redundant columns of index, student_id, sleep_time, and wake_time'
3. for 'student.isnull().sum() for data containing null values'
4. for 'student.isnull().sum() for final_test and attendance_rate’
5. for 'For calculating the skewness of the numerical data'
6. for 'For split the data into features and target labels'
7. for 'For Log-transform the skewed features, check'
8. for 'For features_log_transformed '
9. for 'For For train_test_split data and One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies() and For calculating Naives Bayes using TP, FP, TN, FN'

In the student_MLPiple.py file, there are six pipelines involved:

1. for 'Pipeline 1: Get predictions on the first 300 training samples(X_train)'
2. for 'Pipeline 2: Results, accuracy, fbeta_score on three models of SVC, Decision Tree Classifier, Random Forest Classifier'
3. for 'Printing out the values for 1%, 10% and 100%'
4. for 'Use gridsearch to fit and predict accuracy and fbeta scores'
5. for 'Fit, train and obtain top 5 feature importances features on the chosen model'

e. Overview of key findings from the EDA conducted in Task 1 and the choices made in the pipeline based on these findings, particularly any feature engineering
For Tasks 1 overview of the finding can be found at Implementing Data Exploration from 1 to 11. These 11 steps are briefing described as follows after importing necessary libraries and packages: 
To import the necessary libraries and packages to execute codes, 'sgqalchemy' and its package "create_engine", pandas and seaborn as well as matplotlib are imported. Import Scipy to check for skewness as well as numpy for calculations.

![image](https://github.com/Okja88/student/assets/79552371/4948c2d4-a550-4ae0-92ee-0235fdaabcd4)

In Data Exploration 1, data from file 'score' is read using read_sql, from the database engine that has been created using create_engine from sqlalchemy. First/last 5 rows of data is and can be visible and displayed using student.head/ student.tail. The result is the displaying of column names and their corresponding rows of data.

![image](https://github.com/Okja88/student/assets/79552371/e9768af1-bde9-4736-b304-e34897ac92b3)

In Data Exploration 2, quick overview of the dataset with statistics generated with details such as count, unique, top, frequencies, measures of central tendencies and dispersions, quartiles, as well as minimum and maximum is possible with student.describe() executed.

![image](https://github.com/Okja88/student/assets/79552371/b739b427-d435-4f18-83b4-28ff2988f1ff)

In Data Exploration 3, we will note the three different data types(int64, object,float64) by executing student.dtypes(). Key findings are: 

![image](https://github.com/Okja88/student/assets/79552371/d48ec3f1-bc27-48e2-8606-1c62abafe5d7)

In Data Exploration 4, column names are then derived:

![image](https://github.com/Okja88/student/assets/79552371/ef3832f8-f24b-40a8-a541-33d8a34e76f3)

In Data Exploration 5, discreet continuous variables can be visually checked through using Seaborn's .hist(). 

![image](https://github.com/Okja88/student/assets/79552371/5b503e4b-ec7e-43b4-aa59-50a516ce2f6a)

In Data Exploration 6 , a random check on a column to confirm there are indeed outliers(presented as points outside the whiskers) present in particular pair "age' and "attendance_rate" through the use of boxplot. Just to get a random example.

![image](https://github.com/Okja88/student/assets/79552371/9e4fe03c-f598-4441-98f0-7f1cd1fd4858)

In Data Exploration 7, Seaborn's pairplot is graphically see the relationships of pairs between the features columns. 
In Data Exploration 8 and 9, Student.shape() and student.count() will notify the number of data in rows, hinting missing data. (15900, 18) is the shape. Pointing to possibility of missing/null data in final_test and attendance_rate, the following is derived from .count():

![image](https://github.com/Okja88/student/assets/79552371/328b5bf5-51b4-4823-8b7a-6510f05ec57d)

In Data Exploration 10, duplicates function is called to check for duplication of data resulting in 19 rows of duplicate data from the (0,18) result returned.
Implementing Data Exploration 11 will yield the total number of records, total number students obtaining final test results of up to 50.00, as well as more than 50.00 and also the percentage of individuals obtaining less than 50.00. The results are as follow:

![image](https://github.com/Okja88/student/assets/79552371/41dda6f3-17ff-488b-b0cf-a265b04705da)

f. Described how the features in the dataset are processed (summarized in a table)

![image](https://github.com/Okja88/student/assets/79552371/590f6cd0-473f-4fee-a66b-009327c585b0)
![image](https://github.com/Okja88/student/assets/79552371/51ad0b81-0078-426e-9c3b-8f2f8697ecd1)
![image](https://github.com/Okja88/student/assets/79552371/8e80a5c4-c5a0-4dd8-b704-a63211520798)
![image](https://github.com/Okja88/student/assets/79552371/9ccc3a70-a575-41d9-8bde-89985c2cfd3b)

g. Explanation of your choice of models
Supervised Learning Model Application
The following are some of the supervised learning models that are currently available in scikit-learn that I will use:
•	Support Vector Machines (SVM)
•	Decision Tree Classifier
•	Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)

SVM performs relatively well when there is a clear margin of separation. It is also more effective when there are higher dimensions involved. SVM Model is also memory efficient. IT works well when the number of dimensions is higher than the samples numbers. Complex problems can be easily solved using the appropriate Kernel functions. However, one of the drawbacks of SVM Algorithm is that it relatively does not function well with large datasets. And when there are many overlapping target classes, i.e., more presence of noise. It underperforms when training samples are more than the number of features in data points. It also has the disadvantage of no probabilistic data (only gives us the resultant classes) as the data points are moved to either above or below the hyperplane. SVM has hyperparameters C and gamma which are difficult to fine tune. For my case, SVM can be used for the classification task of donating or not donating. Also, there are many dimensions/features involved in our dataset. Since our dataset is not relatively large, it should be suitable to deploy SVM algorithm. Furthermore, there is no overlapping in our target classes, it is easier to execute once Kernel function is properly tuned.

Decision Tree Algorithm is a supervised machine learning method that constructs a tree-like model to make decisions by recursively partitioning the input data based on feature attributes. Each internal node represents a decision rule based on a specific feature, leading to subsequent branches or leaves that signify the predicted outcome. Its advantages include interpretability, as decision trees are easy to visualize and understand, and the ability to handle both numerical and categorical data. They require minimal data preprocessing and are robust against outliers. However, Decision Trees are prone to overfitting, especially with complex datasets, which can lead to poor generalization. Additionally, they can be sensitive to small variations in the data and may struggle with capturing certain relationships between features.

Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting) The Ensemble methods of combining Adaptive Boosting on a Random Forest model (essentially a collection of Decision Trees model), where the weights are re-assigned to each instance, higher weights assigned to incorrectly classified instances, has the real-life application in the detection of drug sensitivity of a medication. The Gradient Boosting Algorithm is added to predict both the continuous target variable (as a Regressor) and the target variable (as a Classifier) where the cost function Mean Square Error (MSE) is used as a Regressor, or the cost function Log Loss is used as a classifier. Ensemble methods has the advantage of improved accuracy and performance over a single model especially for complex and noisy problems. By using different subsets and features of data using Bagging, trade-offs are achieved by balancing the bias and variance, hence it reduces overfitting or underfitting. Additionally, this algorithm can handle different types of data using classification, regression, clustering by using different base model and aggregation methods. However, the algorithm’s disadvantage is that it is computationally expensive as it includes the training and storing of different models to combine the outputs. It is also difficult to interpret and explain the abstraction and aggregations. If the base model is too weak or strong, it can often result to underfitting or overfitting. This Ensemble Methods is suitable for our “donating or not” case as it can improve our accuracy over a chosen single model both as a classifier or regressor. It has the advantage to prevent overfitting or underfitting especially for complex and noisy problems. Trade-offs between bias and variance is also achieved. Since our dataset is not large, it is suitable for use.
Results: Out of the three models above, Random Forest Classifier is the best algorithm to use as it has the highest F-test score for training set, with ('f_test': 0.9844086254691977). Although Random Forest Classifier took the longest in terms of the prediction time used ('train_time': 0.09034609794616699), the duration taken is many times over Decision Tree algorithm's duration('train_time': 0.015583992004394531), Its prediction time ('pred_time': 0.03123641014099121) is also longer when compared to Decision Tree Classifier('pred_time': 0.0) as the latter Classifier took no time at all to predict.

h. Evaluation of the models developed. Any metrics used in the evaluation should also be explained.
All three models are initialized based on samples of 1%, 10% and 100%. The results on the number of samples are collected from the learners. 

![image](https://github.com/Okja88/student/assets/79552371/240c5449-b1fa-40ba-9b1a-82e12952b0c9)

Based on the above, RandomForest Classifier is chosen to fine tune with more parameters, including random state, ‘gini’ criterion, using cross_validation of 4 fold. But first, we initialized the classifier, followed by performing a gridsearchCV():

# Initialize the classifier
# oob_score SET True to get inner-CrossValidation, oob can only be set to True if bootstrap is True
# using all processes to run in parellel by setting n_jobs to -1
clf_A = RandomForestClassifier(n_jobs = -1, random_state = 2, criterion='gini', min_samples_split=2, bootstrap=True, oob_score = True)

# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
# with n_jobs=1 it uses 100% of the cpu of one of the cores. Each process is run in a different core. with n_jobs = -1, it uses all the cores for parellel processing
# verbose=0 to show nothing (silent), verbose=1 will show animated progress bar, verbose=2 will state number of epoch
# cv(cross validation is set to 4-fold, instead of the default 5Fold, (Stratified)KFold) if the estimator is a classifier and y is either binary or multiclass
grid_obj = GridSearchCV(estimator = clf_A, param_grid = parameters, cv = 4, n_jobs = -1, verbose = 1, scoring = scorer)

![image](https://github.com/Okja88/student/assets/79552371/15fec356-4e64-425c-8d9c-0ba291688146)

Because the train time take by Random Forest Algorithm is the most amongst all the three models, Top 5 feature importance can be extracted by using .features_importances_. These features importances are then standardized. Hence saving us time the next time we would use any similar set of data. We would then use these top 5 features importances to determine if our student is weak.
These are the feature importance:

![image](https://github.com/Okja88/student/assets/79552371/75e434f5-c24f-4cfb-8450-c83b9e0d6de9)

Next, we select the most important features to doublecheck since with less features required to train, the training time and prediction time is much lower now. We now get Accuracy and fBeta scores of 1 based on this subset of data using this block of codes:

![image](https://github.com/Okja88/student/assets/79552371/1f715378-3229-46c4-a047-1392f0555943)

We have the following final results:

![image](https://github.com/Okja88/student/assets/79552371/400de35d-be57-407c-baba-b4b9607fc413)

















