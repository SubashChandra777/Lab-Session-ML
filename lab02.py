import numpy as np
import pandas as pd
def read_Data(filename):
    data = pd.read_excel(filename)
    data.dropna(how='all',axis=1,inplace=True)
    selected_Data=data[['Candies (#)','Mangoes (Kg)','Milk Packets (#)']]
    X=selected_Data.values.tolist()
    selected_Output=data['Payment (Rs)']
    Y=selected_Output.values.tolist()
    return np.array(X),np.array(Y)
def calculate_rank(Matrix):
    Feature_matrix=np.array(Matrix)
    return np.linalg.matrix_rank(Feature_matrix)
def Cost_of_each_product(Features,Output):
    inv=np.linalg.pinv(Features)
    ans=inv @ Output
    return list(ans.round(2))
filename='Lab Session Data.xlsx'
Features,Output=read_Data(filename)
print(Features)
print(Output)
rankOfMatrix=calculate_rank(Features)
print(f"rank of the feature matrix is {rankOfMatrix}")
ans=Cost_of_each_product(Features,Output)
print(f"Cost of each candy is {ans[0]}")
print(f"Cost of each mango is {ans[1]}")
print(f"Cost of each milk packet is {ans[2]}")