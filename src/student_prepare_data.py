# Programing Information
# PROGRAMMER: NATHAN ONG KEE WEE
# DATE CREATED : 15-08-2023
# LAST REVISED DATE : 21-08-2023
# PURPOSE : Prepare the data through cleaning, formatting, and restructuring, report results executing the following below:

# Preparing the Data

# Before utilizing data for machine learning algorithms, it requires preprocessing, which involves cleaning, formatting, and restructuring.
# This dataset has both duplicate data and missing entries, but adjustments are needed for specific features too.
# This preprocessing significantly enhances the effectiveness and predictive capability of various learning algorithms.
# It is also a good practise to have hygiene data for data analytics.

# The following will be implemented:
# In Implementation: Preparing the Data 1, duplicates are dropped through executing student.drop_duplicates() function.
# In Implementation: Preparing the Data 2, columns names such as 'index', 'student_id', 'sleep_time', and 'wake_time' are considered redundant, hence dropped using student.drop(<column name>, axis =1).
# In Implementation: Preparing the Data 3, null data is checked using student.isnull().sum() resulting in the 'final_test' and 'attendance_rate" highlighted as having null data.
# In Implementation: Preparing the Data 4, these null data from the two mentioned columns are dropped.
# In Implementation: Preparing the Data 5, numerical data are checked for skewness using skew(student['n_male'], axis=0, bias=True).
# In Implementation: Preparing the Data 6 the raw data is split into features and target labels.
# In Implementation: Preparing the Data 7, the skewed features are log-transformed and verified.
# In Implementation: Preparing the Data 8, Import sklearn.preprocessing.StandardScaler is used to normalise skewed data of 'n_male', 'n_female', 'number_of_siblings', 'age', 'hours_per_week' , and 'attendance_rate'.
# In Implementation: Preparing the Data 9, non-numerical data is transformed using One-hot encoding of the 'features_log_minmax_transform' data using pandas.get_dummies().
# In Implementation: Preparing the Data 10, we separate the data using this "from sklearn.model_selection import train_test_split".
# In my final Implementation: Preparing the Data 11, Naive Bayes will be  used as our base case to compare, hence I will calculate specificity and accuracy but not precision and recall.

# Import numpy, pandas
import numpy as np
import pandas as pd

# Import scipy
from scipy.stats import skew

# Import the engine from the SQLAlchemy
import sqlalchemy as sqlalchemy

# Import data from file student_eda.py
from student_eda import student

# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Import argparse
import argparse

# Use argparse Expected Call with <> indicating expected user input:
#       python student_eda.py --dir <directory with student data>
#   Example call:
#       python student_eda.py --dir student_ML/ --studentTable score.db

# Create Argument Parser object named parser
parser = argparse.ArgumentParser(description= "Execute the various data preparation of cleaning, transforming, normalising")
parser.usage =  "Type the command: python student.prepare_data.py -v(erbose) <#>" \
                "# - 1 for 'student.drop_duplicates()'" \
                "# - 2 for 'student.drop() for redundant columns of index, student_id, sleep_time, and wake_time'" \
                "# - 3 for 'student.isnull().sum() for data containing null values'" \
                "# - 4 for 'student.isnull().sum() for final_test and attendance_rate'" \
                "# - 5 for 'For calculating the skewness of the numerical data'" \
                "# - 6 for 'For split the data into features and target labels'" \
                "# - 7 for 'For Log-transform the skewed features, check'" \
                "# - 8 for 'For features_log_transformed'" \
                "# - 9 for 'For For train_test_split data and One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies() and For calculating Naives Bayes using TP, FP, TN, FN'" \


# Argument 1: Add a path to the folder where the directory  is student_ML, file is student_eda.py
parser.add_argument('-v', '--verbose', type = int, help ='execute data preparation of cleaning, transforming and normalising', choices=[1,2,3,4,5,6,7,8,9,10,11])

# Assigns variable in_args to parse_args()
in_args = parser.parse_args()

# Access values of Arguments by printing them
# Implementation: Preparing the Data 1
# To drop duplicates
if in_args.verbose == 1:
    # Create the connection to the engine from the sqlite file
    dbEngine = sqlalchemy.create_engine('sqlite://C:Users/Nathan/PycharmProjects/student_ML/score.db')
    student = pd.read_sql('select * from score', dbEngine)
    student = student.drop_duplicates()
    print(student)
# Implementation: Preparing the Data 2
# Drop the columns named 'index', 'student_id', 'sleep_time', and 'wake_time'
elif in_args.verbose == 2:
    # Create the connection to the engine from the sqlite file
    dbEngine = sqlalchemy.create_engine('sqlite://C:Users/Nathan/PycharmProjects/student_ML/score.db')
    student = pd.read_sql('select * from score', dbEngine)
    student = student.drop('index', axis=1)
    student = student.drop('student_id', axis=1)
    student = student.drop('sleep_time', axis=1)
    student = student.drop('wake_time', axis=1)

# Implementation: Preparing the Data 3
# Find null values, must put sum(), if not it will return Null or otherwise
elif in_args.verbose == 3:
    # Create the connection to the engine from the sqlite file
    dbEngine = sqlalchemy.create_engine('sqlite://C:Users/Nathan/PycharmProjects/student_ML/score.db')
    student = pd.read_sql('select * from score', dbEngine)
    print(student.isnull().sum())

# Implementation: Preparing the Data 4
# Drop the missing values of 'final_test' and 'attendance_rate
elif in_args.verbose == 4:
    # Create the connection to the engine from the sqlite file
    dbEngine = sqlalchemy.create_engine('sqlite://C:Users/Nathan/PycharmProjects/student_ML/score.db')
    student = pd.read_sql('select * from score', dbEngine)
    student = student.dropna()
    print(student.count())
    # Check any null data
    print(student.isnull().sum())

# Implementation: Preparing the Data 5
# To calculate the skewness of the numerical data, and print the result
elif in_args.verbose == 5:
    # Create the connection to the engine from the sqlite file
    dbEngine = sqlalchemy.create_engine('sqlite://C:Users/Nathan/PycharmProjects/student_ML/score.db')
    student = pd.read_sql('select * from score', dbEngine)
    print("n_males has a skewness of:", skew(student['n_male'], axis=0, bias=True))
    print("n_females has a skewness of:", skew(student['n_female'], axis=0, bias=True))
    print("number_of_siblings has a skewness of:", skew(student['number_of_siblings'], axis=0, bias=True))
    print("final_test has a skewness of:", skew(student['final_test'], axis=0, bias=True))
    print("age has a skewness of:", skew(student['age'], axis=0, bias=True))
    print("hours_per_week has a skewness of:", skew(student['hours_per_week'], axis=0, bias=True))
    print("attendance_rate has a skewness of:", skew(student['attendance_rate'], axis=0, bias=True))

# Implementation: Preparing the Data 6
# Split the data into features and target labels
elif in_args.verbose == 6:
    # Import the engine from the SQLAlchemy
    import sqlalchemy
    # Create the connection to the engine from the sqlite file
    dbEngine = sqlalchemy.create_engine('sqlite://C:Users/Nathan/PycharmProjects/student_ML/score.db')
    student = pd.read_sql('select * from score', dbEngine)
    finaltest_raw = student['final_test']
    print(finaltest_raw)
    features_raw = student.drop('final_test', axis=1)
    print(features_raw)

# Implementation: Preparing the Data 7
# Log-transform the skewed features, check #and visualise features_log_transformed
elif in_args.verbose == 7:
    # Create the connection to the engine from the sqlite file
    dbEngine = sqlalchemy.create_engine('sqlite://C:Users/Nathan/PycharmProjects/student_ML/score.db')
    student = pd.read_sql('select * from score', dbEngine)
    skewed = ['n_male', 'n_female', 'number_of_siblings', 'age', 'hours_per_week', 'attendance_rate']
    features_raw = student.drop('final_test', axis=1)
    features_log_transformed = pd.DataFrame(data=features_raw)
    features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))
    # Checking log data is transformed
    print(features_log_transformed)

# Implementation: Preparing the Data 8
# Import sklearn.preprocessing.StandardScaler
elif in_args.verbose == 8:
    # Create the connection to the engine from the sqlite file
    dbEngine = sqlalchemy.create_engine('sqlite://C:Users/Nathan/PycharmProjects/student_ML/score.db')
    student = pd.read_sql('select * from score', dbEngine)
    from sklearn.preprocessing import MinMaxScaler
    # Initialize a scaler, then apply it to the features
    scaler = MinMaxScaler()  # default=(0, 1)
    numerical = ['n_male', 'n_female', 'number_of_siblings', 'age', 'hours_per_week', 'attendance_rate']
    features_raw = student.drop('final_test', axis=1)
    features_log_transformed = pd.DataFrame(data=features_raw)
    features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))
    features_log_minmax_transform = pd.DataFrame(data=features_log_transformed)
    features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])
    # Show an example of a record with scaling applied
    print(features_log_minmax_transform.head(n=5))
    features_final = pd.get_dummies(student, columns=['direct_admission', 'CCA', 'learning_style', 'gender', 'tuition',

# Implementation: Preparing the Data 9
# One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
elif in_args.verbose == 9:
    # Import the engine from the SQLAlchemy
    #import sqlalchemy as sqlalchemy
    # Import train_test_split from sklearn
    #from sklearn.model_selection import train_test_split
    # Import numpy
    #import numpy as np
    # Create the connection to the engine from the sqlite file
    dbEngine = sqlalchemy.create_engine('sqlite://C:Users/Nathan/PycharmProjects/student_ML/score.db')
    student = pd.read_sql('select * from score', dbEngine)
    features_final = pd.get_dummies(student, columns=['direct_admission', 'CCA', 'learning_style', 'gender', 'tuition',
                                                      'mode_of_transport', 'bag_color'])
    # Encode the 'final_test' data to numerical values
    final_test = finaltest_raw.apply(lambda x: 1 if x > 50.0 else 0)
    print(final_test)
    # Print the number of features after one-hot encoding
    encoded = list(features_final.columns)
    print("{} total features after one-hot encoding.".format(len(encoded)))
    # Uncomment the following line to see the encoded feature names
    print(encoded)
    # Display features_final after one hot encoding
    print(features_final.head(n=5))

    # Split the 'features_final' and 'final_test' data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_final,
                                                        final_test,
                                                        test_size=0.2,
                                                        random_state=0)
    # Show the results of the split
    print("Training set has {} samples.".format(X_train.shape[0]))
    print("Testing set has {} samples.".format(X_test.shape[0]))

    # Calculate specificity and accuracy but not precision and recall
    #    TP = 0 # No predicted positives in the naive case
    #    FP = 0 # No predicted positives in the naive case
    #    TN = np.sum(final_test) # Counting the ones as this is the naive case. Note that 'final_test' is the 'finaltest_raw' data
    #    encoded to numerical values done in the data preprocessing step.
    #    FN = final_test.count() - TN # Specific to the naive case


    TP = 0
    FP = 0
    TN = np.sum(final_test == 0)
    FN = final_test.count() - TN

    # Accuracy is the fraction of predictions our model got right out of all the predictions.
    # This means that we sum the number of predictions correctly predicted as Positive (TP) or correctly predicted as Negative (TN) and divide it by all types of predictions, both correct (TP, TN) and incorrect (FP, FN).TP / (TP + FN)
    recall = 0  # TP / ( TP + FN)
    precision = 0  # (TP)/(TP + FP)
    specificity = TN / (FP + TN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)  # TP and FP are both zeros, we are left with TN / TN + FN

    # Ignore calculating F-score using the formula above for beta = 0.5 and correct values for precision and recall.
    # fscore = (1+0.5**2) * (precision * recall)/ (((0.5**2) * precision) + recall)

    # Print the results
    print("Naive Predictor: [Specificity score: {:.4f}, Accuracy score: {:.4f}]".format(specificity, accuracy))

def student_prepare():
    # Create the connection to the engine from the sqlite file
    dbEngine = sqlalchemy.create_engine('sqlite://C:Users/Nathan/PycharmProjects/student_ML/score.db')
    student = pd.read_sql('select * from score', dbEngine)
    # Implementation: Preparing the Data 1
    # To drop duplicates
    student = student.drop_duplicates()
    student.head()

    # Implementation: Preparing the Data 2
    # Drop the columns named 'index', 'student_id', 'sleep_time', and 'wake_time'
    student = student.drop('index', axis=1)
    student = student.drop('student_id', axis=1)
    student = student.drop('sleep_time', axis=1)
    student = student.drop('wake_time', axis=1)

    # Check if "index", "student_id", "sleep_time", and "wake_time" is dropped
    print(student)

    # Implementation: Preparing the Data 3
    # Find null values, must put sum(), if not it will return Null or otherwise
    print(student.isnull().sum())

    # Implementation: Preparing the Data 4
    # Drop the missing values of 'final_test' and 'attendance_rate'
    student = student.dropna()
    print(student.count())

    # Check any null data
    print(student.isnull().sum())

    # Implementation: Preparing the Data 5
    # To calculate the skewness of the numerical data, and print the result
    print("n_males has a skewness of:", skew(student['n_male'], axis=0, bias=True))
    print("n_females has a skewness of:", skew(student['n_female'], axis=0, bias=True))
    print("number_of_siblings has a skewness of:", skew(student['number_of_siblings'], axis=0, bias=True))
    print("final_test has a skewness of:", skew(student['final_test'], axis=0, bias=True))
    print("age has a skewness of:", skew(student['age'], axis=0, bias=True))
    print("hours_per_week has a skewness of:", skew(student['hours_per_week'], axis=0, bias=True))
    print("attendance_rate has a skewness of:", skew(student['attendance_rate'], axis=0, bias=True))

    # Implementation: Preparing the Data 6
    # Split the data into features and target labels
    finaltest_raw = student['final_test']
    print(finaltest_raw)
    features_raw = student.drop('final_test', axis = 1)
    print(features_raw)

    # Implementation: Preparing the Data 7
    # Log-transform the skewed features, check #and visualise features_log_transformed
    skewed = ['n_male', 'n_female', 'number_of_siblings','age', 'hours_per_week', 'attendance_rate']
    features_log_transformed = pd.DataFrame(data = features_raw)
    features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))
    # Checking log data is transformed
    print(features_log_transformed)

    # Implementation: Preparing the Data 8
    # Initialize a scaler, then apply it to the features
    scaler = MinMaxScaler() # default=(0, 1)
    numerical = ['n_male', 'n_female', 'number_of_siblings', 'age', 'hours_per_week' , 'attendance_rate']
    features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
    features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])
    # Show an example of a record with scaling applied
    print(features_log_minmax_transform.head(n = 5))

    # Implementation: Preparing the Data 9
    # One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
    features_final = pd.get_dummies(student, columns=['direct_admission','CCA','learning_style','gender','tuition','mode_of_transport','bag_color'])
    # Encode the 'final_test' data to numerical values
    final_test = finaltest_raw.apply(lambda x: 1 if x > 50.0 else 0)
    print(final_test)
    #Print the number of features after one-hot encoding
    encoded = list(features_final.columns)
    print("{} total features after one-hot encoding.".format(len(encoded)))
    # Uncomment the following line to see the encoded feature names
    print(encoded)
    # Display features_final after one hot encoding
    print(features_final.head(n = 5))

    # Implementation: Preparing the Data 10
    # Split the 'features_final' and 'final_test' data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_final,
                                                    final_test,
                                                    test_size = 0.2,
                                                    random_state = 0)
    # Show the results of the split
    print("Training set has {} samples.".format(X_train.shape[0]))
    print("Testing set has {} samples.".format(X_test.shape[0]))

    # Implementation: Preparing the Data 11
    # Calculate specificity and accuracy but not precision and recall
    '''
    TP = 0 # No predicted positives in the naive case
    FP = 0 # No predicted positives in the naive case

    TN = np.sum(final_test) # Counting the ones as this is the naive case. Note that 'final_test' is the 'finaltest_raw' data 
    encoded to numerical values done in the data preprocessing step.
    FN = final_test.count() - TN # Specific to the naive case
    '''

    TP = 0
    FP = 0
    TN = np.sum(final_test == 0)
    FN = final_test.count() - TN

    # Accuracy is the fraction of predictions our model got right out of all the predictions.
    # This means that we sum the number of predictions correctly predicted as Positive (TP) or correctly predicted as Negative (TN) and divide it by all types of predictions, both correct (TP, TN) and incorrect (FP, FN).TP / (TP + FN)
    recall = 0 # TP / ( TP + FN)
    precision = 0 #(TP)/(TP + FP)
    specificity = TN / (FP + TN)
    accuracy = ( TP + TN) / ( TP + TN + FP + FN) # TP and FP are both zeros, we are left with TN / TN + FN

    # Ignore calculating F-score using the formula above for beta = 0.5 and correct values for precision and recall.
    #fscore = (1+0.5**2) * (precision * recall)/ (((0.5**2) * precision) + recall)

    # Print the results
    print("Naive Predictor: [Specificity score: {:.4f}, Accuracy score: {:.4f}]".format(specificity, accuracy))

student_prepare()
