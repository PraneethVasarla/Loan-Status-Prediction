import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics



df = pd.read_csv('loan_train.csv')
print(df.describe())
print(df['Property_Area'].value_counts())

df['ApplicantIncome'].hist(bins=50)
plt.show()
df.boxplot(column='ApplicantIncome')
plt.show()
df.boxplot(column='ApplicantIncome',by='Education')
plt.show()
df['LoanAmount'].hist(bins=50)
plt.show()
df.boxplot(column='LoanAmount')
plt.show()

temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values='Loan_Status',index='Credit_History',aggfunc=lambda x:x.map({'Y':1,'N':0}).mean())
print('Frequency chart for Credit history')
print(temp1)

print('\nProbability of getting loan for each credit history class:')
print(temp2)


fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')


ax2 = fig.add_subplot(122)
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")
temp2.plot(kind = 'bar')
plt.show()

temp3 = df['Married'].value_counts(ascending=True)
temp4 = df.pivot_table(values='Loan_Status',index='Married',aggfunc=lambda x:x.map({'Y':1,'N':0}).mean())
print('Frequency chart for Marital status')
print(temp3)
print('\nProbability of getting loan according to marital status')
print(temp4)

temp3.plot(kind='bar',title='Marital Status')
plt.show()
temp4.plot(kind='bar',title='Marriage vs loan probability')
plt.show()

temp5 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
temp5.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
plt.show()

temp6 = pd.crosstab(df['Married'], df['Loan_Status'])
temp6.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
plt.show()

temp7 = df['Property_Area'].value_counts(ascending=True)
temp8 = df.pivot_table(values='Loan_Status',index='Property_Area',aggfunc=lambda x:x.map({'Y':1,'N':0}).mean())
print('Frequency chart for Property_Area')
print(temp7)
print('\nProbability of getting loan according to Property_Area')
print(temp8)
temp7.plot(kind='bar',title='Property_Area')
plt.show()
temp8.plot(kind='bar',title='Property_Area vs loan probability')
plt.show()


df.boxplot(column='LoanAmount',by=['Education','Self_Employed'])
plt.show()

print('Missing values in Education: '+str(df.Education.isnull().sum()))
print('Missing values in self employed: '+str(df.Self_Employed.isnull().sum()))
print(df.Self_Employed.value_counts())
df['Self_Employed'].fillna('No',inplace=True)
print('Missing values in self employed after imputaion: '+str(df.Self_Employed.isnull().sum()))


table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)

def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]
# Replace missing values
print('before imputing loan amount:\n '+str(df.apply(lambda x: sum(x.isnull()),axis=0)))
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)
print('After imputing loan amount:\n '+str(df.apply(lambda x: sum(x.isnull()),axis=0)))

df['LoanAmount'].hist(bins=20)
plt.title('Before applying log to loan amount to nullify extreme values')
plt.show()

df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)
plt.title('After applying log to loan amount nullify extreme values')
plt.show()

#One intuition can be that some applicants have lower income but strong support Co-applicants. So it might be a good idea to combine both incomes as total income and take a log transformation of the same.

df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome'].hist(bins=20)
plt.title('Total income before applying log to treat extreme values')
plt.show()

df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins=20)
plt.title('Total income after applying log to treat extreme values')
plt.show()

###### BUILDING THE PREDICTIVE MODEL #######

df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

print('Total missing values: \n'+str(df.apply(lambda x:x.isnull().sum(),axis=0)))

cat_vars = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()

for i in cat_vars:
    df[i] = le.fit_transform(df[i])

#print(df.dtypes)
def classification_model(model,data,predictors,outcome):
    model.fit(data[predictors], data[outcome])
    predictions = model.predict(data[predictors])
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print(accuracy*100)
    kf = KFold(data.shape[0],n_folds=5)
    error = []
    for train, test in kf:
        # Filter training data
        train_predictors = (data[predictors].iloc[train, :])

        # The target we're using to train the algorithm.
        train_target = data[outcome].iloc[train]

        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)

        # Record error from each cross-validation run
        error.append(model.score(data[predictors].iloc[test, :], data[outcome].iloc[test]))

    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

    # Fit the model again so that it can be refered outside the function:
    model.fit(data[predictors], data[outcome])

#logistc regression model
outcome_var = 'Loan_Status'
lr = LogisticRegression()
predictor_var = ['Credit_History']
classification_model(lr, df,predictor_var,outcome_var)

#decision tree
dt = DecisionTreeClassifier()
predictor_var = ['Credit_History','Gender','Married','Education']
classification_model(dt, df,predictor_var,outcome_var)

#random forest
rf = RandomForestClassifier(n_estimators=100)
predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'LoanAmount_log','TotalIncome_log']
classification_model(rf, df,predictor_var,outcome_var)

#Create a series with feature importances:
featimp = pd.Series(rf.feature_importances_, index=predictor_var).sort_values(ascending=False)
print (featimp)

model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History','Dependents','Property_Area']
classification_model(model, df,predictor_var,outcome_var)