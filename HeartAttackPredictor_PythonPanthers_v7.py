#############################################################################################
# Team: PYTHON PANTHERS                                                                     #
# Project Name: Heart Disease Analysis and Prediction                                       #
# Course Instructor: Radha Chawla                                                           #
# Team Members:                                                                             #
#              1. Saima Naz                   -   8745332                                   #
#              2. Sri Hari Varma Valivarthi   -   8739371                                   #
#              3. Sai Manoj Varre             -   8740395                                   #
#              4. Hasan Mahamud Rana          -   8732852                                   #
#              5. Naveenreddy Majjiga         -   8751999                                   #
#                                                                                           #
# Date Written: July-06-2021                                                                #
# Description: This program performs the Heart condition analysis of the patients and       #
#              predicts if any of the patients are at high risk of Heart Attack.            #
#              This program trains the below 7 classification techniques and determines     #
#              the Machine learning model with High Accuracy                                #
#              1. LogisticRegression Classfication                                          #
#              2. KNeighborsClassifier Classfication                                        #
#              3. SVC Classfication                                                         #
#              4. GaussianNB Classfication                                                  #
#              5. RandomForestClassifier Classfication                                      #
#              6. DecisionTreeClassifier Classfication                                      #
#              7. xgboost Classfication                                                     #
#                                                                                           #
#############################################################################################

from ssl import Options
import justpy as jp
from justpy.htmlcomponents import Option
import pandas
import os
from pandas.core.dtypes.common import classes

# To split the Input data to train the classification techniques
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# For plots
import matplotlib.pyplot as plt
import seaborn as sb

# Set the Current Working Directory
# os.chdir("D:/Conestoga/Heart_Analysis_and_Prediction/")
os.chdir("D:/Education/Semester 1/21S - Sec2/PROG 8420 Programming for Big Data (Radha)/Projects/Predictive Analysis/Input_Files")

# Load the Input Excel spreadsheet to a data frame
Training_ML_File = pandas.read_excel("Heart_Disease_Input_Data.xlsx")

Training_ML_File_DF = pandas.DataFrame(Training_ML_File)

# Load the Testing Input Excel spreadsheet  file to a data frame
Patient_Record = pandas.read_excel("Patient_Test_Data.xlsx")

Patient_Record_DF = pandas.DataFrame(Patient_Record)

# Check the sample data from the input file
print(Training_ML_File.head())

# Get the column names of the Training_ML_File using Keys()
Column_Names = Training_ML_File.keys()
print(Column_Names)

# Number of rows in Training_ML_File
Total_Rows = Training_ML_File.shape[0]
print(Total_Rows)

# Number of columns in Training_ML_File
Total_Columns = Training_ML_File.shape[1]
print(Total_Columns)

# Summary of the data available in the Training_ML_File
Training_ML_File.describe()

#############################################################################################
#               MACHINE LEARNING USING THE CLASSIFICATION TECHNIQUES                        #
#############################################################################################
#                        DATA SPLIT AND NORMALIZATION                                       #
#############################################################################################

# Assign the first 13 columns values to Source_Data.
Source_Data = Training_ML_File.iloc[0:Total_Rows, 0:Total_Columns-1].values

# Assign the first last columns values to Target.
Target = Training_ML_File.iloc[0:Total_Rows, -1].values

# Split the Machine learning Training File into two sets i.e.  Training set and Test set
Source_Data_train, Source_Data_test, Target_train, Target_test = train_test_split(
    Source_Data, Target, test_size=0.2, random_state=1)

# Normalize the Machine learning Training File Data
Scaler = StandardScaler()
Source_Data_train = Scaler.fit_transform(Source_Data_train)
Source_Data_test = Scaler.transform(Source_Data_test)

#############################################################################################
#                               TRAIN THE CLASSIFICATION MODEL                              #
#############################################################################################
# Train the classification models using the dataframes which are split using the Input data

Model_Accuracy_Dictnry = {}
Model = []
Target_pred = []
Model_Names = []

# This function uses 6 Machine learning models and determines the accuracy rate of the predictions


def train_the_models(n):
    if Model[n] == 0:
        Model[n] = LogisticRegression(random_state=1)  # get instance of model
        Model_Names.append('Logistic Regression')
    elif Model[n] == 1:
        Model[n] = KNeighborsClassifier()  # get instance of model
        Model_Names.append('KNN')
    elif Model[n] == 2:
        Model[n] = SVC(random_state=1)  # get instance of model
        Model_Names.append('SVC')
    elif Model[n] == 3:
        Model[n] = GaussianNB()  # get instance of model
        Model_Names.append('GaussianNB')
    elif Model[n] == 4:
        Model[n] = RandomForestClassifier(
            random_state=1)  # get instance of model
        Model_Names.append('Random Forest')
    elif Model[n] == 5:
        Model[n] = XGBClassifier(random_state=1)  # get instance of model
        Model_Names.append('XGBoost')
    else:
        Model[n] = DecisionTreeClassifier(
            random_state=1)  # get instance of model
        Model_Names.append('Decision Tree')

    Model[n].fit(Source_Data_train, Target_train)  # Train/ Fit model

    Target_pred.append(Model[n].predict(Source_Data_test))  # get y predictions

    Model_Accuracy = accuracy_score(Target_test, Target_pred[n])
    Model_Accuracy_Dictnry[Model[n]] = Model_Accuracy


# Call the machine learning classification models function
Number_of_models = 7

for n in range(Number_of_models):
    Model.append(n)
    train_the_models(Model[n])

#############################################################################################
#                       CHOOSING HIGH ACCURACY MODEL                                        #
#############################################################################################
# Check which Model has the highest ACCURACY percentage
Max_Key = max(Model_Accuracy_Dictnry, key=Model_Accuracy_Dictnry.get)
print(Max_Key)

# Check the Accuracy percentage of the model
all_values = Model_Accuracy_Dictnry.values()
max_value = max(all_values)
Accuracy_list = []
Accuracy_List = list(all_values)

print(max_value)

# Get the Index position of the highest Accuracy percentage
Target_pred_pos = Model.index(Max_Key)

#############################################################################################
#                      COMPARE CLASSIFICATION REPORTS FOR ALL THE MODELS                    #
#############################################################################################

Classification_Dictionary = {}
List_Classification = []
Number_of_models = 6
for n in range(Number_of_models):
    Classification_Dictionary = classification_report(
        Target_test, Target_pred[n], output_dict=True)

    List_Classification.append([Classification_Dictionary["0"]["precision"], Classification_Dictionary["0"]
                               ["recall"], Classification_Dictionary["0"]["f1-score"], Classification_Dictionary["0"]["support"]])
    List_Classification.append([Classification_Dictionary["1"]["precision"], Classification_Dictionary["1"]
                               ["recall"], Classification_Dictionary["1"]["f1-score"], Classification_Dictionary["1"]["support"]])
    List_Classification.append([Classification_Dictionary["macro avg"]["precision"], Classification_Dictionary["macro avg"]
                               ["recall"], Classification_Dictionary["macro avg"]["f1-score"], Classification_Dictionary["macro avg"]["support"]])
    List_Classification.append([Classification_Dictionary["weighted avg"]["precision"], Classification_Dictionary["weighted avg"]
                               ["recall"], Classification_Dictionary["weighted avg"]["f1-score"], Classification_Dictionary["weighted avg"]["support"]])

    List_Classification_DF = pandas.DataFrame(List_Classification)

List_Classification_DF.columns = ['PRECISION', 'RECALL', 'F1-SCORE', 'SUPPORT']

List_Classification_DF1 = List_Classification_DF

Classification_Type = [' - Positive', ' - Negative',
                       ' - macro avg', ' - weighted avg']
Classification_Type_Num = 0
Model_Name_Num = 0

for n in range(len(List_Classification_DF)):

    if Classification_Type_Num < 4:
        Row_Name = Model_Names[Model_Name_Num] + \
            Classification_Type[Classification_Type_Num]
        List_Classification_DF1 = List_Classification_DF1.rename(index={
                                                                 n: Row_Name})
        List_Classification_DF2 = List_Classification_DF1
        Classification_Type_Num = Classification_Type_Num+1

        if Classification_Type_Num == 4:
            Classification_Type_Num = 0
            Model_Name_Num = Model_Name_Num+1

List_Classification_DF2.to_excel(
    'Classification_Data.xlsx', sheet_name="Model Comparison", index=True)

#############################################################################################
#                      COMPARE ACCURACY AND CONFUSION MATRIX FOR ALL THE MODELS             #
#############################################################################################

Conf_Matrix_List = []
Conf_Matrix_List_Temp = []

Number_of_models = 6
for n in range(Number_of_models):

    Conf_Matrix = confusion_matrix(Target_test, Target_pred[n])
    Conf_Matrix.flatten().tolist()
    Conf_Matrix_List_Temp.append(Conf_Matrix.flatten().tolist())
    Conf_Matrix_List = Conf_Matrix_List_Temp
    Conf_Matrix_List[n].append(Accuracy_List[n])              # add accuracy
    Conf_Matrix_List_DF = pandas.DataFrame(Conf_Matrix_List)

for n in range(Number_of_models):

    Conf_Matrix_List_DF = Conf_Matrix_List_DF.rename(index={n: Model_Names[n]})

Conf_Matrix_List_DF.columns = [
    'True Positive', 'False Positive', 'True Negative', 'False Negative', 'Accuracy']

Conf_Matrix_List_DF.to_excel(
    'Confusion_Matrix.xlsx', sheet_name="True Pos-Neg & False Pos-Neg", index=True)

#############################################################################################
#                      COMPARE PREDICTION WITH ACTUAL RESULT                                #
#############################################################################################

# Compare the model predicted values with the actual values and save the result to excel sheet

Number_of_models = 6
Compare_List = []
Compare_List.append(Target_test)

for n in range(Number_of_models):
    Target_pred = Model[n].predict(Source_Data_test)
    Compare_List.append(Target_pred)

Compare_List_DF = pandas.DataFrame(Compare_List)

Number_of_models = 6
Index_Num = 0
for n in range(Number_of_models):

    if n == 0:
        Compare_List_DF = Compare_List_DF.rename(index={n: 'Actual_Value'})
        Index_Num = Index_Num + 1
        Compare_List_DF = Compare_List_DF.rename(
            index={Index_Num: Model_Names[n]})
    else:
        Index_Num = Index_Num+1
        Compare_List_DF = Compare_List_DF.rename(
            index={Index_Num: Model_Names[n]})

Compare_List_DF = Compare_List_DF.transpose()
Compare_List_DF.to_excel('Model_Comparison.xlsx',
                         sheet_name="Actual VS Prediction", index=True)

#############################################################################################
#                                TESTING MODULE                                             #
# Prediction testing to check if a patient has risk of Heart attack.                        #
# Prediction is performed Based on the highest accuracy model                               #
# Max_Key value has the model with the highest percentage of the accuracy                   #
#############################################################################################

Heart_Condition_Predictor_List = []

# Copy the Testing file i.e. Patient_Health_Record to a List
Patient_Record_List_row = Patient_Record.values.tolist(
)                         # Copy Only Values to the List
# Copy Only Column names to the List
Patient_Record_List_Col = Patient_Record.columns.values.tolist()
High_Risk = ['High Risk']
No_Risk = ['No Risk']

for i in range(len(Patient_Record_List_row)):

    Predict_HeartCondition = Max_Key.predict(
        Scaler.transform([Patient_Record_List_row[i]]))

    Heart_Condition_Predictor_List.append(Patient_Record_List_row[i])

    if Predict_HeartCondition == 1:
        Heart_Condition_Predictor_List[i].extend(High_Risk)
    else:
        Heart_Condition_Predictor_List[i].extend(No_Risk)

# Copy the List Output to Dataframe
Heart_Condition_Predictor_DF = pandas.DataFrame(Heart_Condition_Predictor_List)

# Give column names to the Dataframe
Heart_Condition_Predictor_DF.columns = ['Age', 'Sex', 'Chest_Pain_Type', 'Resting_BP', 'Cholesterol', 'Fasting_Blood_Sugar', 'Resting ECG',
                                        'Max_Heart_Rate', 'Exercise_Angina', 'ST_Segment_Depression', 'Heart_Rate_Slope', 'Fluoroscopy_Vessels', 'Thalassemia', 'Heart_Attack_Prediction']

# Generate Output Excel Spreadsheet file with Prediction results
Heart_Condition_Predictor_DF.to_excel(
    'Patient_Heart_Condition_Predictor.xlsx', index=False)


with pandas.ExcelWriter('Heart_Attack_Analysis.xlsx', engine='xlsxwriter') as writer:
    Training_ML_File_DF.to_excel(writer, sheet_name='Input_Data', index=True)
    Compare_List_DF.to_excel(
        writer, sheet_name='Actual VS Prediction', index=True)
    List_Classification_DF2.to_excel(
        writer, sheet_name='Model Comparison', index=True)
    Conf_Matrix_List_DF.to_excel(
        writer, sheet_name='True Pos-Neg & False Pos-Neg', index=True)
    Patient_Record_DF.to_excel(
        writer, sheet_name='Patient_Record_Testing_File', index=True)
    Heart_Condition_Predictor_DF.to_excel(
        writer, sheet_name='Patient_Record_Testing_Results', index=True)


#############################################################################################
#                                      Exploratory Analysis                                 #
#############################################################################################

# Create correlation and create heatmap

check_correlation = Training_ML_File.corr()
plt.subplots(figsize=(15, 10))
heatmapplot = sb.heatmap(check_correlation, xticklabels=check_correlation.columns,
                         yticklabels=check_correlation.columns,
                         annot=True,
                         cmap="RdPu")
# cmap="YlGnBu")

heatmapplot.figure.savefig('heatmap.png')

#############################################################################################

# Pair Plot

PairPlotData = Training_ML_File[[
    'Age', 'Resting_BP', 'Cholesterol', 'Max_Heart_Rate', 'ST_Segment_Depression']]
pairplot = sb.pairplot(PairPlotData)

pairplot.savefig('Pairplot.png')

#############################################################################################

Accuracy_def = """
{
    chart: {
        type: 'column'
    },
    title: {
        text: '7 classification techniques'
    },
    subtitle: {
        text: 'Accuracy'
    },
    accessibility: {
        announceNewData: {
            enabled: true
        }
    },
    xAxis: {
        type: 'category'
    },
    yAxis: {
        title: {
            text: 'Accuracy'
        }

    },
    legend: {
        enabled: false
    },
    plotOptions: {
        series: {
            borderWidth: 0,
            dataLabels: {
                enabled: true,
                format: '{point.y:.1f}%'
            }
        }
    },

    tooltip: {
        headerFormat: '<span style="font-size:11px">{series.name}</span><br>',
        pointFormat: '<span style="color:{point.color}">{point.name}</span>: <b>{point.y:.2f}%</b><br/>'
    },

    series: [
        {
            name: "Accuracy",
            colorByPoint: true,
            data: [
                {
                    name: "Model Name",
                    y: 0
                },
                {
                    name: "Model Name",
                    y: 0
                },
                {
                    name: "Model Name",
                    y: 0
                },
                {
                    name: "Model Name",
                    y: 0
                },
                {
                    name: "Model Name",
                    y: 0
                },
                {
                    name: "Model Name",
                    y: 0
                },
                {
                    name: "Model Name",
                    y: 0
                },
            ]
        }
    ]
}
"""

Compare_deff = """
{
    chart: {
        type: 'bar'
    },
    title: {
        text: 'Compare Accuracy and Confusion Matrix for All The Models'
    },
    subtitle: {
        text: 'Source:'
    },
    xAxis: {
        categories: [],
        title: {
            text: null
        }
    },
    yAxis: {
        min: 0,
        title: {
            text: 'Comparison',
            align: 'high'
        },
        labels: {
            overflow: 'justify'
        }
    },
    tooltip: {
        valueSuffix: ' '
    },
    plotOptions: {
        bar: {
            dataLabels: {
                enabled: true
            }
        }
    },
    legend: {
        layout: 'vertical',
        align: 'right',
        verticalAlign: 'top',
        x: -40,
        y: 80,
        floating: true,
        borderWidth: 1,
        backgroundColor: '#FFFFFF',
        shadow: true
    },
    credits: {
        enabled: false
    },
    series: [{
        name: '',
        data: []
    }, {
        name: '',
        data: []
    }, {
        name: '',
        data: []
    }, {
        name: '',
        data: []
    }]
}
"""


def app():

    wp = jp.QuasarPage()
#--------------------------------------------------------------------------------------------------#
    d1 = jp.QDiv(classes="q-gutter-lg row justify-center", a=wp)
    hc1 = jp.HighCharts(a=d1, options=Accuracy_def)

    Multiplied_Accuracy = [element * 100 for element in Accuracy_List]

    for i in range(len(Model_Names)):
        hc1.options.series[0].data[i].name = Model_Names[i]
        hc1.options.series[0].data[i].y = Multiplied_Accuracy[i]

#--------------------------------------------------------------------------------------------------#

    d2 = jp.QDiv(classes="q-gutter-lg row justify-center", a=wp)
    hc2 = jp.HighCharts(a=d2, options=Compare_deff)

    hc2.options.xAxis.categories = list(Conf_Matrix_List_DF.index)

    hc2.options.series = [{'name': v1, 'data': [v2 for v2 in Conf_Matrix_List_DF[v1]]}
                          for v1 in Conf_Matrix_List_DF.columns]

#--------------------------------------------------------------------------------------------------#

    d3 = jp.QDiv(classes="q-gutter-lg row justify-center", a=wp)
    hc3 = jp.Matplotlib(a=d3)

#--------------------------------------------------------------------------------------------------#

    d4 = jp.QDiv(classes="q-gutter-lg row justify-center", a=wp)
    hc4 = jp.HighCharts(a=d4, options=Compare_deff)

    hc4.options.xAxis.categories = list(List_Classification_DF2.index)

    hc4.options.series = [{'name': v1, 'data': [v2 for v2 in List_Classification_DF2[v1]]}
                          for v1 in List_Classification_DF2.columns]

    return wp


jp.justpy(app)


# def plot_test():
#     wp = jp.WebPage()
#     plt.plot([0, 1, 4, 9], marker='*', markersize=20, markeredgecolor='red')
#     plt.title('Matplotlib Example')
#     plt.xlabel('x data')
#     plt.ylabel('y data')
#     jp.Matplotlib(a=wp)

#     # If memory may be an issue, don't forget to close figures not in use anymore
#     plt.close()
#     return wp


# jp.justpy(plot_test)

#############################################################################################
#                                      END OF PROGRAM                                       #
#############################################################################################
