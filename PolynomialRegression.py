from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

#importing data
data = pd.read_csv()

#retrieving the columns as dataframes
x = data.iloc[:,0:1] #age column
y = data.iloc[:,1:2] #height columns

#from dataframe to array conversion
X = x.values
Y = y.values

#linear regression
lin_reg1 = LinearRegression()
lin_reg1.fit(X, Y)

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)

#linear regression
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, Y)

#visualisaiton
plt.subplot(1,2,1)
plt.scatter(X, Y, color = "r")
plt.plot(X, lin_reg1.predict(X), color="b")
plt.title("Linear Regression")
plt.grid(True)

plt.subplot(1,2,2)
plt.scatter(X, Y, color="r")
plt.plot(X, lin_reg2.predict(poly_reg.transform(X)), color="b")
plt.title("Polynomial Regression")
plt.grid(True)