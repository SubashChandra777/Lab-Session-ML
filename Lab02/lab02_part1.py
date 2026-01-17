import numpy as np
import pandas as pd
#Loading the data from the file
def load_data(file_name):
    return pd.read_excel(file_name,usecols='A:E')
#To read Features and output values,to store in array
def read_Data(data):
    selected_Data=data[['Candies (#)','Mangoes (Kg)','Milk Packets (#)']]
    X=selected_Data.values.tolist()
    selected_Output=data['Payment (Rs)']
    Y=selected_Output.values.tolist()
    return np.array(X),np.array(Y)
#To calculate the rank of the Feature Vector with numpy package
def calculate_rank(Matrix):
    Feature_matrix=np.array(Matrix)
    return np.linalg.matrix_rank(Feature_matrix)
#To calculate the Cost of each product using pseudo inverse in numpy package
def Cost_of_each_product(Features,Output):
    inv=np.linalg.pinv(Features)
    ans=inv @ Output
    return list(ans.round(2))
#Classifer model to categorize customers as RICH if payment is above 200 Rs and others as POOR
def Classification(Output):
    if Output>200:
        return "RICH"
    else:
        return "POOR"
#A1
#Loading data and storing into DataFrame
file_path = 'Lab Session Data.xlsx'
df = load_data(file_path)

#Extract Feature Vector and Output Values Vector
matrix_Feature, vector_Output = read_Data(df)

# Calculate Rank of the Feature Vector
rank = calculate_rank(matrix_Feature)
print(f"Dimensionality (Rank) of the vector space: {rank}")

#Calculate Cost of each product (Model vector X)
# This solves the equation: Y * X = O
costs = Cost_of_each_product(matrix_Feature, vector_Output)
print(f"Cost of Candies, Mangoes, Milk: {costs}")

#Classification of Rich and Poor
#applying your Classification function to the 'Payment (Rs)' column
df['Category'] = df['Payment (Rs)'].apply(Classification)

#A2
# Displaying the Categorised Customers
print(df[['Customer','Payment (Rs)', 'Category']].to_string(index=False))
