import numpy as np
def MatrixMultiplication(mat1,mat2):
    #Get the dimensions of the both matrices
    row_1,col_1 = mat1.shape
    row_2,col_2 = mat2.shape
    #Comparing no of columns of 1st matrix to the no of rows of 2nd matrix if it's not equal then it's not possible
    if col_1 != row_2:
        return "Matrix Multiplication Failed"
    else:
        #performing matrix multiplication using @ operator as per numpy module
        return mat1 @ mat2

def CountVowelsAndConsonants(str):
    vowels = ['A', 'E', 'I', 'O', 'U']
    count = 0
    #iterating through the character of the input string
    for char in str:
        #checking if character is a vowel by converting it to upper case
        if char.upper() in vowels:
            #increasing the count if the character read is a vowel
            count += 1
    #returning vowel count and consonant count,(consonants=length of string - vowel_count)
    return count,len(str)-count

def CommonElements(list1,list2):
    ans=0
    #iterating through the 1st list and checking the existence within the 2nd list
    for i in list1:
        if i in list2:
            #increasing the count if both the elements are equal
            ans+=1
    return ans

def Mean_Mode_Median():
    #generating 100 random integers between 100 and 150 and storing them in numpy array
    list_of_100=np.random.randint(100,151,100)
    #Calculation of mean:sum of observations/no of observations
    n=len(list_of_100)
    mean = sum(list_of_100)/n
    #Calculation of mode using sort and scan method
    list_of_100.sort()
    current_count = 1
    max_count =1
    mode = list_of_100[0]
    #looping from 2nd element to the last element
    for i in range(1,n):
        if list_of_100[i]==list_of_100[i-1]:
            current_count+=1#continue the count if present element equal to previous element
        else:
            current_count=1#reset the count if they aren't the same
        #update mode if the count is greater than the max_count
        if current_count>max_count:
            max_count=current_count
            mode=list_of_100[i]
    #Calculation of Median
    #since no of observations is even(100),median is the average between two middle elements
    median = (list_of_100[n//2 - 1]+list_of_100[n//2])/2
    return mean,mode,median

def Transpose(matrix):
    #returning the transpose of the input matrix using transpose function in the numpy module
    return np.transpose(matrix)


#Q1:Vowels and Consonants
input_str = input("Enter a string: ")
v_count, c_count = CountVowelsAndConsonants(input_str)
print(f"Number of Vowels: {v_count}")
print(f"Number of Consonants: {c_count}")

#Q2:Matrix Multiplication
row_matrix1=int(input("Enter the no of rows for first matrix: "))
col_matrix1=int(input("Enter the no of columns for first matrix: "))
A=[]
for i in range(row_matrix1):
    A.append(list(map(int,input().split())))
matrix1=np.array(A)
B=[]
row_matrix2=int(input("Enter the no of rows for second matrix: "))
col_matrix2=int(input("Enter the no of columns for second matrix: "))
for j in range(row_matrix2):
    B.append(list(map(int,input().split())))
matrix2=np.array(B)
result_mul = MatrixMultiplication(matrix1, matrix2)
print(f"Product of AB: {result_mul}")

#Q3:Common Elements
print("Enter the first list of numbers (separated by space):")
list_1 = list(map(int, input().split()))
print("Enter the second list of numbers (separated by space):")
list_2 = list(map(int, input().split()))
common = CommonElements(list_1, list_2)
print(f"Number of common elements: {common}")

#Q4:Matrix Transpose
row_t = int(input("Enter rows for the matrix to transpose: "))
col_t = int(input("Enter columns for the matrix to transpose: "))
print("Enter matrix entries row by row:")
T_entries = []
for i in range(row_t):
    T_entries.append(list(map(int, input().split())))
matrix_to_transpose = np.array(T_entries)
transposed_matrix = Transpose(matrix_to_transpose)
print(f"Transposed Matrix:{transposed_matrix}")

#Q5:Mean,Mode and Median
mean_val, mode_val, median_val = Mean_Mode_Median()
print(f"Mean: {mean_val}; Mode: {mode_val}; Median: {median_val}")



