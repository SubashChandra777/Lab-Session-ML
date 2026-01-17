import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading the data from the file
def load_data(filename):
    return pd.read_excel(filename,sheet_name="IRCTC Stock Price",usecols='A:I')
#Calculating Mean and Variance using numpy functions
def Calculate_MeanAndVariance(data):
    column_D=data['Price']
    X=np.array(column_D.values)
    return X.mean(),X.var()
#Calculating Mean using Formula
def Own_Mean_and_Variance(data):
    column_D=data['Price']
    sum_of_observations=column_D.values.sum()
    No_of_observations=column_D.shape[0]
    Own_mean=sum_of_observations/No_of_observations#Mean = X1+X2+......Xn/n
    observations_=pow(column_D.values-Own_mean,2)
    Own_variance=observations_.sum()/No_of_observations#Variance = (X1-Mean)^2+(X2-Mean)^2+......(Xn-Mean)^2/n
    return Own_mean,Own_variance
#Comparing Accuracy of the Mean and Variance from the derived formula and from the inbuilt func in numpy
def Compare_Accuracy(own_mean,Own_variance,package_mean,package_variance):
    if np.isclose(own_mean,package_mean) and np.isclose(Own_variance,package_variance):
        return True
    else:
        return False
#Measurng the Complexity of the Fuction by running the function 10 times
def Measure_Complexity(data,function):
    total_time=0
    for i in range(10):
        start_time = time.time()
        function(data)
        end_time = time.time()
        total_time += end_time - start_time
    avg_time = total_time/10
    return avg_time
#Calculating the Mean of the prices on the day of Wednesday
def Get_Wednesdat_mean(data):
    wedenesday_data=data[data['Day']=='Wed']#Sorting data with Wednesday
    return wedenesday_data['Price'].mean()#Calculating Mean using numpy function
#Calculating the Mean of the prices on the month of April
def Get_April_mean(data):
    april_data=data[data['Month']=='Apr']#Sorting data with the month of April
    return april_data['Price'].mean()#Calculating the Mean using numpy function
#Comparing the Sample Mean and the Population Mean
def Compare_mean_of_sample_and_population(sample_mean,population_mean):
    if np.isclose(sample_mean,population_mean):
        return True
    return False
#Calculating the Probability of Loss
def Calculate_Loss_Probability(data):
    clean_chg=data['Chg%'].astype(str).str.replace('%','').astype(float)
    loss_stocks=clean_chg.apply(lambda x:x < 0)
    probability_of_loss=sum(loss_stocks)/len(clean_chg)
    return probability_of_loss
#Calculating the Probability of Profit on Wednesdays
def Calculate_Profit_Probability_Wednesday(data):
    wednesday_data = data[data['Day'] == 'Wed']
    clean_chg = wednesday_data['Chg%'].astype(str).str.replace('%', '').astype(float)
    profit_stocks = clean_chg.apply(lambda x: x > 0)
    probability_of_profit = sum(profit_stocks) / len(data)
    return probability_of_profit
#Calculating the Conditional Probability of Profit on Wednesdays
def Calculate_Conditional_ProfitProbaility_Wednesday(data):
    wednesday_data=data[data['Day']=='Wed']
    clean_chg=wednesday_data['Chg%'].astype(str).str.replace('%','').astype(float)
    profit_stocks=clean_chg.apply(lambda x:x > 0)
    probability_of_profit=sum(profit_stocks)/len(clean_chg)
    return probability_of_profit
#Plotting the Change % with Days
def Plot_Chng_Vs_Day(data):
    plot_data=data.copy()
    plot_data['Chg%_Clean']=plot_data['Chg%'].astype(str).str.replace('%','').astype(float)
    order_ofDays=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    plot_data['Day']=plot_data['Day'].astype(str).str.strip()
    plot_data['Day']=pd.Categorical(plot_data['Day'],categories=order_ofDays,ordered=True)
    plot_data=plot_data.sort_values(by=['Day'])
    plt.figure(figsize=(10,6))
    plt.scatter(plot_data['Day'],plot_data['Chg%_Clean'],color='green',alpha=0.5)
    plt.title('Scatter plot of Chng data against the Day of the week')
    plt.ylabel('Price change Percentage')
    plt.xlabel('Day of the week')
    plt.grid(True)
    plt.show()

#A3
filename = 'Lab Session Data.xlsx'
data = load_data(filename)

#Package and  Own,Mean & Variance calculations
pkg_mean, pkg_var = Calculate_MeanAndVariance(data)
own_mean, own_var = Own_Mean_and_Variance(data)
print(f"Package: Mean={pkg_mean:.4f}, Var={pkg_var:.4f}")
print(f"Own:     Mean={own_mean:.4f}, Var={own_var:.4f}")

if Compare_Accuracy(own_mean, own_var, pkg_mean, pkg_var):
        print("Accuracy Check: PASSED")
else:
        print("Accuracy Check: FAILED")

#Checking the Complexity of package,own  mean and variance
avg_time_pkg = Measure_Complexity(data, Calculate_MeanAndVariance)
avg_time_own = Measure_Complexity(data, Own_Mean_and_Variance)
print(f"Pkg Time: {avg_time_pkg:.6f}s")
print(f"Own Time: {avg_time_own:.6f}s")

#Mean of prices on Wednesdays
print(f"Wednesday Mean: {Get_Wednesdat_mean(data):.4f}")
#Mean of prices in the month of April
print(f"April Mean:     {Get_April_mean(data):.4f}")

#Calculation of Probabilities
prob_loss = Calculate_Loss_Probability(data)#Total loss probabilty
prob_profit_wed = Calculate_Profit_Probability_Wednesday(data)#Probability of Profit on Wednesdays
prob_cond_wed = Calculate_Conditional_ProfitProbaility_Wednesday(data)#Conditional Probability on Wednesdays

print(f"Loss Probability: {prob_loss:.4f}")
print(f"Profit on Wed (Joint): {prob_profit_wed:.4f}")
print(f"Profit given Wed (Conditional): {prob_cond_wed:.4f}")

#Ploting Change % with days
Plot_Chng_Vs_Day(data)