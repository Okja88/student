# Programing Information
# PROGRAMMER: NATHAN ONG KEE WEE
# DATE CREATED : 15-08-2023
# LAST REVISED DATE : 21-08-2023
# PURPOSE : Do exploratory data analysis, report results executing the following below:

# Importing packages and libraries
# To import the necessary libraries and packages to execute codes, 'sgqalchemy' and its package "create_engine", pandas and seaborn as well as matplotlib are imported."""
# Import Scipy to check for skewness as well as numpy for calculations."""

# Reading file using sqlalchemy
# Data from file 'score' is read using read_sql, from the database engine that has been created using create_engine from sqlalchemy."""

# First/last 5 rows of data is and can be visible and displayed using student.head/ student.tail.""""""
# The result is the displaying of column names and their corresponding rows of data in Data Exploration 1.
# Quick overview of the dataset with statistics generated with details such as count, unique, top, frequencies, measures of central tendencies and dispersions, quartiles, as well as minimum and maximum is possible with student.describe() executed in Data Exploration 2.
# In Data Exploration 3, I would have noted the three different data types(int64, object,float64) by executing student.dtypes().
# Column names are then derived in Data Exploration 4. Discreet continuous variables can be visually checked through using Seaborn's .hist() in Data Exploration 5.
# A random check on a column to confirm there are indeed outliers(presented as points outside the whiskers) present in particular pair "age' and "attendance_rate" in Data Exploration 6 through the use of boxplot.
# In Data Exploration 7, Seaborn's pairplot is graphically see the relationships of pairs between the features columns.
# Student.shape() and student.count() in Data Exploration 8 and 9 will notify the number of data in rows, hinting missing data.
# In Data Exploration 10, duplicates function is called to check for duplication of data resulting in 19 rows of duplicate data.
# Implementing Data Exploration 11 will yield the total number of records, total number students obtaining final test results of up to 50.00, as well as more than 50.00 and also the percentage of Percentage of individuals obtaining less than 50.00.

# Import pandas, seaborn, matplotlib.pyplot
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)

# Import the engine from the SQLAlchemy
import sqlalchemy

# Import argparse
import argparse

# Use argparse Expected Call with <> indicating expected user input:
#       python student_eda.py --dir <directory with student data>
#   Example call:
#       python student_eda.py --dir student_ML/ --studentTable score.db

# Create Argument Parser object named parser
parser = argparse.ArgumentParser(description= "Find the various exploratory analysis")
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

# Argument 1: Add a path to the folder where the directory  is student_ML, file is student_eda.py
parser.add_argument('-v', '--verbose', type = int, help ='execute student exploratory data analysis', choices=[1,2,3,4,5,6,7,8,9,10])

# Assigns variable in_args to parse_args()
in_args = parser.parse_args()

# Access values of Arguments by printing them
# Implementation: Data Exploration 1
# Read data with pandas and visually verify using either .head() method
if in_args.verbose == 1:
    # Create the connection to the engine from the sqlite file
    dbEngine = sqlalchemy.create_engine('sqlite://C:Users/Nathan/PycharmProjects/student_ML/score.db')
    student = pd.read_sql('select * from score', dbEngine)
    print("student.head will return: ", student.head())
# Implementation: Data Exploration 2
elif in_args.verbose == 2:
    # Create the connection to the engine from the sqlite file
    dbEngine = sqlalchemy.create_engine('sqlite://C:Users/Nathan/PycharmProjects/student_ML/score.db')
    student = pd.read_sql('select * from score', dbEngine)
    print("student.describe will return: ", student.describe(include="all"))
# Implementation: Data Exploration 3
elif in_args.verbose == 3:
    # Create the connection to the engine from the sqlite file
    dbEngine = sqlalchemy.create_engine('sqlite://C:Users/Nathan/PycharmProjects/student_ML/score.db')
    student = pd.read_sql('select * from score', dbEngine)
    print("student.dtypes will return: ", student.dtypes)
# Implementation: Data Exploration 4
elif in_args.verbose == 4:
    # Create the connection to the engine from the sqlite file
    dbEngine = sqlalchemy.create_engine('sqlite://C:Users/Nathan/PycharmProjects/student_ML/score.db')
    student = pd.read_sql('select * from score', dbEngine)
    print("student.columns will return: ", student.columns)
# Implementation: Data Exploration 5
# Using Seaborn with .hist(figsize)
elif in_args.verbose == 5:
    # Create the connection to the engine from the sqlite file
    dbEngine = sqlalchemy.create_engine('sqlite://C:Users/Nathan/PycharmProjects/student_ML/score.db')
    student = pd.read_sql('select * from score', dbEngine)
    print("student.hist will return: ", student.hist(figsize=(20,30)))
# Implementation: Data Exploration 6
elif in_args.verbose == 6:
    # Create the connection to the engine from the sqlite file
    dbEngine = sqlalchemy.create_engine('sqlite://C:Users/Nathan/PycharmProjects/student_ML/score.db')
    student = pd.read_sql('select * from score', dbEngine)
    print("sns.boxplot will return: ",sns.boxplot(x="age", y="attendance_rate", data=student))
# Implementation: Data Exploration 7
elif in_args.verbose == 7:
    # Create the connection to the engine from the sqlite file
    dbEngine = sqlalchemy.create_engine('sqlite://C:Users/Nathan/PycharmProjects/student_ML/score.db')
    student = pd.read_sql('select * from score', dbEngine)
    print("sns.pairplot(student) will return: ", sns.pairplot(student))
# Implementation: Data Exploration 8
elif in_args.verbose == 8:
    # Create the connection to the engine from the sqlite file
    dbEngine = sqlalchemy.create_engine('sqlite://C:Users/Nathan/PycharmProjects/student_ML/score.db')
    student = pd.read_sql('select * from score', dbEngine)
    print("student.shape will return : ", student.shape)
# Implementation: Data Exploration 9
# Count the number of rows before removing the duplicates
elif in_args.verbose == 9:
    # Create the connection to the engine from the sqlite file
    dbEngine = sqlalchemy.create_engine('sqlite://C:Users/Nathan/PycharmProjects/student_ML/score.db')
    student = pd.read_sql('select * from score', dbEngine)
    print("student.count will return: ", student.count())
# Implementation: Data Exploration 10
# Rows containing duplicate data
elif in_args.verbose == 10:
    # Create the connection to the engine from the sqlite file
    dbEngine = sqlalchemy.create_engine('sqlite://C:Users/Nathan/PycharmProjects/student_ML/score.db')
    student = pd.read_sql('select * from score', dbEngine)
    duplicated_student = student[student.duplicated()]
    print("No. of duplicated rows: ", duplicated_student.shape)
# Implementation: Data Exploration 11
# Total number of records, Individuals obtaining more than 50.00,
# Individuals obtaining at most 50.00, Percentage of individuals obtaining less than 50.00
elif in_args.verbose == 11:
    dbEngine = sqlalchemy.create_engine('sqlite://C:Users/Nathan/PycharmProjects/student_ML/score.db')
    student = pd.read_sql('select * from score', dbEngine)
    df = pd.DataFrame(student)
    n_records = len(df.index)

    # Number of records where individual's final_test is more than 50.00
    n_greater_50 = len(df.loc[df['final_test'] >= 50.00])

    # Number of records where individual's final_test is at most 50.00
    n_at_most_50 = len(df.loc[df['final_test'] < 50.00])

    # Percentage of individuals whose final_test is more than 50.00
    greater_percent = (n_at_most_50 / (n_greater_50 + n_at_most_50)) * 100
    # Print the results
    print("Total number of records: {}".format(n_records))
    print("Individuals obtaining more than 50.00: {}".format(n_greater_50))
    print("Individuals obtaining at most 50.00: {}".format(n_at_most_50))
    print("Percentage of individuals obtaining less than 50.00: {}%".format(greater_percent))

else:
    dbEngine = sqlalchemy.create_engine('sqlite://C:Users/Nathan/PycharmProjects/student_ML/score.db')
    student = pd.read_sql('select * from score', dbEngine)
    print("student.head will return: ", student.head())
    print("student.describe will return: ", student.describe(include="all"))
    print("student.dtypes will return: ", student.dtypes)
    print("student.columns will return: ", student.columns)
    print("student.hist will return: ", student.hist(figsize=(20, 30)))
    print("sns.boxplot will return: ", sns.boxplot(x="age", y="attendance_rate", data=student))
    print("sns.pairplot(student) will return: ", sns.pairplot(student))
    print("student.shape will return : ", student.shape)
    print("student.count will return: ", student.count())
    duplicated_student = student[student.duplicated()]
    print("No. of duplicated rows: ", duplicated_student.shape)
    df = pd.DataFrame(student)
    n_records = len(df.index)

    # Number of records where individual's final_test is more than 50.00
    n_greater_50 = len(df.loc[df['final_test'] >= 50.00])

    # Number of records where individual's final_test is at most 50.00
    n_at_most_50 = len(df.loc[df['final_test'] < 50.00])

    # Percentage of individuals whose final_test is more than 50.00
    greater_percent = (n_at_most_50 / (n_greater_50 + n_at_most_50)) * 100

    # Print the results
    print("Total number of records: {}".format(n_records))
    print("Individuals obtaining more than 50.00: {}".format(n_greater_50))
    print("Individuals obtaining at most 50.00: {}".format(n_at_most_50))
    print("Percentage of individuals obtaining less than 50.00: {}%".format(greater_percent))

def student_explore():
    # Create the connection to the engine from the sqlite file
    dbEngine = sqlalchemy.create_engine('sqlite://C:Users/Nathan/PycharmProjects/student_ML/score.db')
    student = pd.read_sql('select * from score', dbEngine)
    # Implementation: Data Exploration 1
    # Read data with pandas and visually verify using .head() method
    student.head()

    # Implementation: Data Exploration 2
    student.describe(include="all")

    # Implementation: Data Exploration 3
    print(student.dtypes)

    # Implementation: Data Exploration 4
    print(student.columns)

    # Implementation: Data Exploration 5
    # Using Seaborn with .hist(figsize)
    student.hist(figsize=(20,30))

    # Implementation: Data Exploration 6
    sns.boxplot(x="age", y="attendance_rate", data=student)

    # Implementation: Data Exploration 7
    print(sns.pairplot(student))
    # Implementation: Data Exploration 8
    print(student.shape)

    # Implementation: Data Exploration 9
    # Count the number of rows before removing the duplicates
    print(student.count())

    # Implementation: Data Exploration 10
    # Rows containing duplicate data
    duplicated_student = student[student.duplicated()]
    print("No. of duplicated rows: ", duplicated_student.shape)

    # Implementation: Data Exploration 11
    # Total number of records
    df=pd.DataFrame(student)
    n_records = len(df.index)

    # Number of records where individual's final_test is more than 50.00
    n_greater_50 = len(df.loc[df['final_test'] >= 50.00])

    # Number of records where individual's final_test is at most 50.00
    n_at_most_50 = len(df.loc[df['final_test'] < 50.00])

    # Percentage of individuals whose final_test is more than 50.00
    greater_percent = (n_at_most_50/(n_greater_50 + n_at_most_50))* 100

    # Print the results
    print("Total number of records: {}".format(n_records))
    print("Individuals obtaining more than 50.00: {}".format(n_greater_50))
    print("Individuals obtaining at most 50.00: {}".format(n_at_most_50))
    print("Percentage of individuals obtaining less than 50.00: {}%".format(greater_percent))
student_explore()
