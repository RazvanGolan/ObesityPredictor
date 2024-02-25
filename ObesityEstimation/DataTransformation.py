import numpy as np
import pandas as pd

# fetch dataset, I want to use only some attributes from the dataset
data = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv", sep = ",")
selected_attributes = ["Gender", "Age", "Height", "Weight", "family_history_with_overweight", "FAVC", "SMOKE",
                       "CH2O", "SCC", "FAF", "CALC", "NObeyesdad"]
data = data[selected_attributes]

# Define mappings for categorical attributes
gender_mapping = {'Female': 0, 'Male': 1}
family_history_mapping = {'no': 0, 'yes': 1}
yes_no_mapping = {'no': 0, 'yes': 1}
calc_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
obesity_mapping = {'Insufficient_Weight' : 0, 'Normal_Weight' : 1, 'Overweight_Level_I' : 2, 'Overweight_Level_II' : 3,
                   'Obesity_Type_I' : 4, 'Obesity_Type_II' : 5, 'Obesity_Type_III' : 6}

data['Gender'] = data['Gender'].map(gender_mapping)
data['family_history_with_overweight'] = data['family_history_with_overweight'].map(family_history_mapping)
data['SMOKE'] = data['SMOKE'].map(yes_no_mapping)
data['FAVC'] = data['FAVC'].map(yes_no_mapping)
data['SCC'] = data['SCC'].map(yes_no_mapping)
data['CALC'] = data['CALC'].map(calc_mapping)
data['Height'] = data['Height'] * 100 #transform into centimeters
data['NObeyesdad'] = data['NObeyesdad'].map(obesity_mapping)


# Display the transformed data
# print(data.head())


predict = "NObeyesdad"

X = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])


def get_parameters():
    return X, y
