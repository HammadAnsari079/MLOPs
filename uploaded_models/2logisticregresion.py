import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
x = np.array([[25,30,1],[29,20,1],[35,31,0],[40,10,1],[20,5,0],[32,2,1],[30,25,0],[50,40,1],[23,39,0],[43,29,1]])
y = np.array([1,1,0,1,0,1,0,1,0,1])
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 42)
model = LogisticRegression()
model.fit(x_train,y_train)
accuracy=model.score(x_test,y_test)
print("The accuracy of the model is :",accuracy)
user_Age = float(input("Enter the age of the customer: "))
user_time_spent = float(input("Enter the amount of time spent om the website: "))
user_cart = int(input("Enter tthe likeliness of the user to put the products in the cart: "))
user_data = np.array([[user_Age,user_time_spent,user_cart]])
modelprediction=model.predict(user_data)
print("The ouptut for the user: ",modelprediction)