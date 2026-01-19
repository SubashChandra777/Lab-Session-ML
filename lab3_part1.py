import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Dot_Product(A,B):
    vector_A=np.array(A)
    vector_B=np.array(B)
    return np.dot(vector_A,vector_B)
def Length_of_Vector(A):
    vector_A=np.array(A)
    return np.linalg.norm(vector_A)
def Calculate_Mean(A):
    return np.mean(A)
def Calculate_Variance(A):
    return np.var(A)
def Calculate_Std(A):
    return np.std(A)
def Mean_for_EachClass(data):
    for col in data.columns:
        data[col].add(Calculate_Mean(data[col]))
def Variance_for_EachClass(data):
    for col in data.columns:
        data[col].add(Calculate_Variance(data[col]))
def Distance_Between_MeanVectors(A,B):
    return np.linalg.norm(A - B)
def Density_Pattern(data):
    x=np.histogram()
    plt.hist(x)
    plt.show()
def Calculate__Minkowski(Feature1, Feature2):
    x=[]
    for i,j in zip(Feature1, Feature2):
        for p in range(10):
            x.append(pow(pow(abs(i-j),p),1/p))
    plt.plot(x)
    plt.show()






