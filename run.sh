Instructions for executing the pipeline and modifying any parameters
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
