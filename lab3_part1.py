import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

def Load_Data():
    filename = "Banknote_Authentication.xlsx"
    df = pd.read_excel(filename)
    Features= df.iloc[:,:-1].values
    Target= df.iloc[:,-1].values
    return Features,Target
#Calculates the dot product between Two vectors
def Dot_Product(A,B):
    vector_A=np.array(A)
    vector_B=np.array(B)
    return np.dot(vector_A,vector_B)#returs the dot product of two vectors calculated from the numpy module

#Calculates the length of the vector(Norm of the vector)
def Length_of_Vector(A):
    vector_A=np.array(A)
    return np.linalg.norm(vector_A)#returs the length of the vector calculated from the numpy module

#Calculates the Mean for Every class(Class Centroid)
def Calculate_ClassCentroid(A,B):
    classes=np.unique(B)
    centoids={}
    for c in classes:
        #Filtering rows belonging to class c
        class_data=A[B==c]
        #Calculating mean across all classes
        centoids[c]=np.mean(class_data,axis=0)
    return centoids # returns a dictionary of centroids for each class with format {class lable:centroid}

#Calculates the Standard Deviation of every class(Class Spread)
def Calculate_ClassSpread(A,B):
    classes=np.unique(B)
    spreads={}
    for c in classes:
        # Filtering rows belonging to class c
        class_data=A[B==c]
        # Calculating standard deviation across all classes
        spreads[c]=np.std(class_data,axis=0)
    return spreads#returns dictionary of standard deviation for each class with format {class lable:standard deviation}
#Calculating Distance between mean vectors
def Distance_Between_MeanVectors(centroid1,centroid2):
    return np.linalg.norm(centroid1 - centroid2)# Returns the distance between two mean vectors
#Plotting the histogram
def Density_Pattern(Feature_data,feature_name):
    hist_data=np.histogram(Feature_data)
    plt.hist(hist_data, bins=20,)
    plt.title(f"Density Pattern:{feature_name}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
#Calculating the Minkowski distance of order p
def Calculate_Minkowski(Feature1, Feature2):
    distance_values=[]
    for p in range(1,11):
        diff=np.abs(np.array(Feature1)-np.array(Feature2))
        distance=np.power(np.sum(np.power(diff,p)),1/p)
        distance_values.append(distance)
    return distance_values
#Plotting the minkowski Distance from order 1 to 11
def Plot_Minkowski_distances(distance_values):
    plt.plot(range(1,11),distance_values,'go-',label='Minkowski Distances')
    plt.title("Minkowski Distances vs p")
    plt.ylabel('Distance')
    plt.xlabel('p value')
    plt.legend()
    plt.show()
#Dividing the datset into test and traim
def Test_And_Train(X,y):
    return train_test_split(X,y,test_size=0.3)
#Classifying the KNN
def KNN_Classifier(X_train,y_train,k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    return neigh
#Calculating the accuracy of the KNN
def Calculate_KNN_Accuracy(model,X_test,y_test):
    return model.score(X_test,y_test)
#Calculating the prediction of the KNN
def Predict_KNN(model,X_test):
    return model.predict(X_test)
def Own_KNN_Classifier(X_train,y_train,test_vector,k):
    distances = []
    #Calculating Euclidean distance to all training points
    for i in range(len(X_train)):
        dist = np.linalg.norm(X_train[i] - test_vector)
        distances.append((dist, y_train[i]))
    #Sorting by distance in ascending order
    distances.sort(key=lambda x: x[0])
    #Getting top k neighbors
    neighbors = distances[:k]
    # Extracting just the labels from the tuple (distance, label)
    neighbor_labels = [label for (dist, label) in neighbors]
    if sum(neighbor_labels) > k/2:
        return 1
    return 0
#Own KNN predictions
def Own_KNN_Predict_Batch(X_train,y_train,X_test,k):
    predictions = []
    for row in X_test:
        predictions.append(Own_KNN_Classifier(X_train,y_train,row,k))
    return np.array(predictions)
#Comparing Own knn model with inbuilt model from sklearn
def Compare_KNN_Classifier(predictions_sklearn,predictions_own_knn):
    mismatches = np.sum(predictions_sklearn != predictions_own_knn)
    total = len(predictions_sklearn)
    match_rate = (total - mismatches) / total
    return mismatches, match_rate
#Plotting the KNN
def Accuracy_KNN_Plot(X_train,y_train,X_test,y_test):
    accuracies = []
    k_range = range(1, 12)
    for k in k_range:
        
        model = KNN_Classifier(X_train, y_train, k)
        accuracies.append(Calculate_KNN_Accuracy(model, X_test, y_test))
    plt.figure()
    plt.plot(k_range, accuracies, marker='s', color='green')
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("KNN Accuracy vs k")
    plt.xticks(k_range)
    plt.show()
#Calculating Performing metrics  with Confusion matrix 
def Performance_Metrics_ConfusionMatrix_TrainAndTest(model,X_train,y_train,X_test,y_test):
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    train_accuracy = accuracy_score(y_train,train_predictions)
    test_accuracy = accuracy_score(y_test,test_predictions)
    confusionMatrix_Train = confusion_matrix(y_train, train_predictions)
    classificationReport_Train = classification_report(y_train, train_predictions)
    confusionMatrix_Test = confusion_matrix(y_test, test_predictions)
    classificationReport_Test = classification_report(y_test, test_predictions)
    accuracy_diff=train_accuracy-test_accuracy
    if train_accuracy < 0.60:
        return "Underfit"
    elif accuracy_diff > 0.10:
        return "Overfit"
    return "Regular"
#Evaluating Confusion Matrix on own
def Own_Confusion_Matrix(y_actual,y_pred):
    TruePositive=0
    TrueNegative=0
    FalsePositive=0
    FalseNegative=0
    for actual,predict in zip(y_actual,y_pred):
        if actual==1 and predict==1:TruePositive+=1
        elif actual==0 and predict==0:TrueNegative+=1
        elif actual==0 and predict==1:FalsePositive+=1
        elif actual==1 and predict==0:FalseNegative+=1
    return np.array([[TruePositive, FalsePositive], [FalseNegative,TruePositive]])
#Evaluating performance metrics on own
def Own_Performance_Metrics(y_actual,y_pred):
    confusion_matrix=Own_Confusion_Matrix(y_actual,y_pred)
    TrueNeg,FalsePos=confusion_matrix[0]
    FalseNeg,TruePos=confusion_matrix[1]

    accuracy = (TruePos+TrueNeg)/(TruePos+TrueNeg+FalsePos+FalseNeg)
    precision=0
    recall=0
    f1=0
    if (TruePos+FalsePos)>0:
     precision = TruePos / (TruePos + FalsePos)
    if (TruePos+FalseNeg)>0:
        recall = TruePos / (TruePos + FalseNeg)
    if (precision+recall)>0:
        f1 = (2*precision*recall)/(precision+recall)
    return accuracy, precision, recall, f1,confusion_matrix
#Calculating Inverse Matrix
def Train_Linear_Classifier(X_train,y_train):
    return np.linalg.pinv(X_train) @ y_train#returns pseudo inverse: W = (X^T X)-1 X^T y
#Prediction of Linear Classifier from the test
def Predict_Linear_Classifier(Linear_Classifier,X_test):
    y_raw=X_test @ Linear_Classifier
    predictions=[]
    for value in y_raw:
        if value>=0.5:
            predictions.append(1)
        else:
            predictions.append(0)
    return np.array(predictions)
#Comparison of KNN with Matrix inversion Technique
def Compare_KNN_with_Matrix_InversionTechnique(knn_model,X_train,y_train,X_test,y_test):
    knn_predictions = knn_model.predict(X_train)
    knn_accuracy = knn_model.score(X_train,y_train)
    linear_classifier=Train_Linear_Classifier(X_train,y_train)
    linear_predictions=Predict_Linear_Classifier(linear_classifier,X_test)
    linear_accuracy=accuracy_score(y_test,linear_predictions)
    if knn_accuracy > linear_accuracy + 0.05:
        return "KNN is better"
    elif linear_accuracy > knn_accuracy + 0.05:
        return "Matrix Inversion is better"
    return "Both models perform similarly"

#Loads the data and creates Feature vector and the target Vector
X, y = Load_Data()

if X is not None:
    print(f"Data Loaded. Shape: {X.shape}")

    #A1
    #All Vector Operations
    v1, v2 = X[0], X[1]
    print(f"Dot Product: {Dot_Product(v1, v2):.2f}")
    print(f"Length of Vector 1: {Length_of_Vector(v1):.2f}")

    #A2
    #Calculations of Class Statistics
    centroids = Calculate_ClassCentroid(X, y)
    spreads = Calculate_ClassSpread(X, y)
    print("Centroids:", list(centroids.keys()))
    print("Spread (Class 0):", spreads[0])
    print("Distance between Class 0 & 1:", Distance_Between_MeanVectors(centroids[0], centroids[1]))

    #A3
    #Plotting of the Density Pattern
    Density_Pattern(X[:, 0], "Variance Feature")

    dists = Calculate_Minkowski(v1, v2)#A4 Calclulations of minkowski distances
    Plot_Minkowski_distances(dists)#A5 plotting of Minkowski distances

    #A6
    #Splitting data set into Test and train
    X_train, X_test, y_train, y_test = Test_And_Train(X, y)
    print(f"Train Size: {X_train.shape[0]}, Test Size: {X_test.shape[0]}")

    #A7 to A9
    k_val = 3
    knn_model = KNN_Classifier(X_train, y_train, k_val)
    acc_sklearn = Calculate_KNN_Accuracy(knn_model, X_test, y_test)
    print(f"Sklearn KNN Accuracy (k={k_val}): {acc_sklearn:.4f}")

    #A10
    #Calssification of KNN using own functions
    preds_sklearn = Predict_KNN(knn_model, X_test)
    preds_own = Own_KNN_Predict_Batch(X_train, y_train, X_test, k_val)
    mismatches, rate = Compare_KNN_Classifier(preds_sklearn, preds_own)
    print(f"Match Rate with Sklearn: {rate * 100:.2f}% ({mismatches} mismatches)")

    #A11
    #Accuracy Plot of KNN
    Accuracy_KNN_Plot(X_train, y_train, X_test, y_test)

    #A12
    #Performance Analysis of the Train And Test
    status = Performance_Metrics_ConfusionMatrix_TrainAndTest(knn_model, X_train, y_train, X_test, y_test)
    print(f"Model Status: {status}")

    #A13
    #Metrics Calculation using own functions
    acc, prec, rec, f1, cm = Own_Performance_Metrics(y_test, preds_own)
    print(f"My Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print("My Confusion Matrix:\n", cm)

    #A14
    #Comparision of KNN with Matrix inversion
    conclusion = Compare_KNN_with_Matrix_InversionTechnique(knn_model, X_test, y_test, X_train, y_train)
    print(f"Conclusion: {conclusion}")
