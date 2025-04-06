# # 2025-04-06 C2W3 lab 01 practice

# for array compuation and loading data
import numpy as np

# building linear regression model
from sklearn.linear_model import LinearRegression #！类
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error #!函数

# for building and training neural networks
import tensorflow as tf 

# custom functions
import utils 

# reduce display precision on numpy arrays
np.set_printoptions(precision=2)

# suppress warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# Load the dataset from the text file
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# data_path = os.path.join(current_dir, 'data', 'data_w3_ex1.csv') #返回上一级目录的data文件夹里的file.txt文件的绝对路径。
# print(os.getcwd())
# data = np.loadtxt(data_path, delimiter=',')
# # data = np.loadtxt('./data/data_w3_ex1.csv', delimiter=',')

# x = data[:,0]
# y = data[:,1]

# # x = x.reshape(-1,1) # Reshape x to be a 2D array with one column
# # y = y.reshape(-1,1) # Reshape y to be a 2D array with one column

# # Convert 1-D arrays into 2-D because the commands later will require it 
# x = np.expand_dims(x, axis=1) # Convert x to a 2D array with one column
# y = np.expand_dims(y, axis=1) # Convert y to a 2D array with one column

# # Plot the entire dataset
# # utils.plot_dataset(x=x, y=y, title="input vs. target")

# # Get 60% of the data for training and 40% for testing
# # put the remaining 40% in the temporaray variables: x_ and y_ 
# x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.4, random_state=1)

# x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.5, random_state=1)

# del x_, y_ # delete the temporary variables to free up memory

# print(f"the shape of the training set (input) is: {x_train.shape}")
# print(f"the shape of the training set (target) is: {y_train.shape}\n")
# print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
# print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")
# print(f"the shape of the test set (input) is: {x_test.shape}")
# print(f"the shape of the test set (target) is: {y_test.shape}")

# # utils.plot_train_cv_test(x_train, y_train, x_cv, y_cv, x_test, y_test, title=" input vs. target")
# # Plot the training, cross-validation, and test sets

# ## Feature Scaling
# scaler_linear = StandardScaler()

# # Compute the mean and standard deviation of the training set
# x_train_scaled = scaler_linear.fit_transform(x_train)

# print(f"Computed mean of the training set: {scaler_linear.mean_.squeeze():.2f}") #! remove the extra dimension
# print(f"Computed standard deviation of the training set: {scaler_linear.scale_.squeeze():.2f}") 

# # plot the results
# # utils.plot_dataset(x=x_train_scaled, y=y_train, title="scaled input vs. target")

# ##! Train the model
# # Initialize the class
# linear_model = LinearRegression()

# # Train the model 
# linear_model.fit(x_train_scaled, y_train)


# ##! Evaluate the model
# # Feed the scaled training set and get the predictions
# yhat = linear_model.predict(x_train_scaled)

# # Use scikit-learn' utility function and divide by 2
# print(f"training MSE (using sklearn function): {mean_squared_error(y_train, yhat) / 2:.2f}")

# # for-loop implementation
# total_squared_error = 0

# for i in range(len(yhat)):
#     squared_error_i = (yhat[i] - y_train[i]) ** 2
#     total_squared_error += squared_error_i
    
# mse = total_squared_error / (2*len(yhat))

# print(f"training MSE (using for-loop): {mse.squeeze():.2f}")

# # Compute over the cross-validation set
# X_cv_scaled = scaler_linear.transform(x_cv) #! transform the cross-validation set using the same scaler
# print(f"Mean used to scale the CV set: {scaler_linear.mean_.squeeze():.2f}")
# print(f"Standard deviation used to scale the CV set: {scaler_linear.scale_.squeeze():.2f}") 

# # Feed the scaled cross-validation set
# yhat = linear_model.predict(X_cv_scaled)
# print(f"Cross validation MSE: {mean_squared_error(y_cv, yhat) / 2:.2f}")

# ## Add polynomial features
# #! Initialize the class to make polynomial features 
# poly = PolynomialFeatures(degree=2, include_bias=False)

# X_train_mapped = poly.fit_transform(x_train)

# print(X_train_mapped[:5]) # x, x^2

# # Instantiate the class
# scaler_poly = StandardScaler()

# X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)

# print(X_train_mapped_scaled[:5]) # x, x^2

# # Initialize the class
# model = LinearRegression()
# model.fit(X_train_mapped_scaled, y_train)

# yhat = model.predict(X_train_mapped_scaled)
# print(f"Training MSE: {mean_squared_error(y_train, yhat) / 2:.2f}")

# X_cv_mapped = poly.transform(x_cv)  # Add the polynomial features to the cross validation set
# X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

# yhat = model.predict(X_cv_mapped_scaled)
# print(f"Cross validation MSE: {mean_squared_error(y_cv, yhat) / 2:.2f}")


# ## Compare different models with different degress
# # Initialize the lists to save the errors 
# train_mses = []# mean squared errors
# cv_mses = []
# models = []
# polys = []
# scalers = []

# for degree in range(1, 11):
    
#     poly = PolynomialFeatures(degree=degree, include_bias=False)
#     X_train_mapped = poly.fit_transform(x_train)
#     polys.append(poly) # save the polynomial features object
    
#     # Scale the training set
#     scaler_poly = StandardScaler()
#     X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
#     scalers.append(scaler_poly)
    
#     # Create and train the model
#     model = LinearRegression()
#     model.fit(X_train_mapped_scaled, y_train)
#     models.append(model) # save the model
    
#     # Compute the training MSE
#     yhat = model.predict(X_train_mapped_scaled)
#     train_mse = mean_squared_error(y_train, yhat) / 2
#     train_mses.append(train_mse)
    
#     # App polynomial features to the cross-validation set
#     X_cv_mapped = poly.transform(x_cv)
#     X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)
    
#     # Compute the cross validation MSE
#     yhat = model.predict(X_cv_mapped_scaled)
#     print(f"Cross validation MSE: {mean_squared_error(y_cv, yhat) / 2:.2f}")
    
# # plot the results
# degrees = range(1,11)
# utils.plot_train_cv_mses(degrees, train_mses, cv_mses, title="degress of polynomial vs. train and CV MSEs")

# degree = np.argmin(cv_mses) + 1
# print(f"lowest CV MSE is found in the model with degree={degree}")

# #! Generalization error by computing the test set's MSE
# X_test_mapped = polys[degree-1].transform(x_test)

# X_test_mapped_scaled = scalers[degree-1].transform(X_test_mapped)

# # Compute the test MSE
# yhat = models[degree-1].predict(X_test_mapped_scaled)
# test_mse = mean_squared_error(y_test, yhat) / 2

# print(f"Training MSE: {train_mses[degree-1]:.2f}")
# print(f"Cross validation MSE: {cv_mses[degree-1]:.2f}")
# print(f"Test MSE: {test_mse:.2f}")

# ## Neural Networks
# # below add polynomial features to see the effect of polynomial features on the neural network
# degree = 1
# poly = PolynomialFeatures(degree=degree, include_bias=False)
# x_train_mapped = poly.fit_transform(x_train)
# x_cv_mapped = poly.transform(x_cv)
# x_test_mapped = poly.transform(x_test)  

# # Build and train the model
# nn_train_mses = []
# nn_cv_mses = []

# # build the models
# nn_models = utils.build_models() 

# # Loop over the models
# for model in nn_models:
    
#     # Setup hte loss and optimizer
#     model.compile(loss='mse',
#                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),)
    
#     print(f"Training {model.name} model")
    
#     # Train the model
#     model.fit(
#         X_train_mapped_scaled, y_train,
#         epochs=300,
#         verbose=0
#     )
    
#     print("Done!\n")

#     # Record the training MSEs
#     yhat = model.predict(X_train_mapped_scaled)
#     train_mse = mean_squared_error(y_train, yhat) / 2
#     nn_train_mses.append(train_mse)
    
#     # Record the cross-validation MSEs
#     yhat = model.predict(X_cv_mapped_scaled)
#     cv_mse = mean_squared_error(y_cv, yhat) / 2
#     nn_cv_mses.append(cv_mse)
    
# # print results
# print("RESULTS:")
# for model_num in range(len(nn_train_mses)):
#     print(
#         f"Model {model_num+1}: Training MSE: {nn_train_mses[model_num]:.2f}, " +
#         f"CV MSE: {nn_cv_mses[model_num]:.2f}"
#     )   
    
# # Select the model with the lowest CV MSE
# model_num = 3

# # Compute the test MSE
# yhat = nn_models[model_num-1].predict(X_test_mapped_scaled)
# test_mse = mean_squared_error(y_test, yhat) / 2

# print(f"Selected Model: {model_num}")
# print(f"Training MSE: {nn_train_mses[model_num-1]:.2f}")
# print(f"Cross Validation MSE: {nn_cv_mses[model_num-1]:.2f}")
# print(f"Test MSE: {test_mse:.2f}")
    
# ##! Classification 
# Load the dataset from the text file
data_path = os.path.join(current_dir, 'data', 'data_w3_ex2.csv') #返回上一级目录的data文件夹里的file.txt文件的绝对路径。
data = np.loadtxt(data_path, delimiter=',')
x_bc = data[:, :-1] # all columns except the last one
y_bc = data[:, -1] # last column

y_bc = np.expand_dims(y_bc, axis=1)
print(f"the shape of the inputs is: {x_bc.shape}")
print(f"the shape of the targets is: {y_bc.shape}")

utils.plot_bc_dataset(x=x_bc, y=y_bc, title="x1 vs. x2")

# from sklearn.model_selection import train_test_split

# Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables.
x_bc_train, x_, y_bc_train, y_ = train_test_split(x_bc, y_bc, test_size=0.40, random_state=1)

# Split the 40% subset above into two: one half for cross validation and the other for the test set
x_bc_cv, x_bc_test, y_bc_cv, y_bc_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

# Delete temporary variables
del x_, y_

print(f"the shape of the training set (input) is: {x_bc_train.shape}")
print(f"the shape of the training set (target) is: {y_bc_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_bc_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_bc_cv.shape}\n")
print(f"the shape of the test set (input) is: {x_bc_test.shape}")
print(f"the shape of the test set (target) is: {y_bc_test.shape}")

from sklearn.model_selection import train_test_split

# Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables.
x_bc_train, x_, y_bc_train, y_ = train_test_split(x_bc, y_bc, test_size=0.40, random_state=1)

# Split the 40% subset above into two: one half for cross validation and the other for the test set
x_bc_cv, x_bc_test, y_bc_cv, y_bc_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

# Delete temporary variables
del x_, y_

print(f"the shape of the training set (input) is: {x_bc_train.shape}")
print(f"the shape of the training set (target) is: {y_bc_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_bc_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_bc_cv.shape}\n")
print(f"the shape of the test set (input) is: {x_bc_test.shape}")
print(f"the shape of the test set (target) is: {y_bc_test.shape}")

# Scale the features
# Initialize the class
scaler_linear = StandardScaler()

# Compute the mean and standard deviation of the training set then transform it
x_bc_train_scaled = scaler_linear.fit_transform(x_bc_train)
x_bc_cv_scaled = scaler_linear.transform(x_bc_cv)
x_bc_test_scaled = scaler_linear.transform(x_bc_test)



# Evaluating the error for classification models
# Sample model output
probabilities = np.array([ 0.2, 0.6, 0.7, 0.3, 0.8])
# Apply a threshold of 0.5 to get the model output
predictions = np.where(probabilities >= 0.5, 1, 0)

ground_truth = np.array([1, 1, 1, 1, 1])

misclassified = 0

# Get number of predictions
num_predictions = len(predictions)

for i in range(num_predictions):
    
    if predictions[i] != ground_truth[i]:
        misclassified += 1
        
# Compute the fraction of data that the model misclassified
fraction_error = misclassified / num_predictions
print(f"probabilities: {probabilities}")
print(f"predictions with threshold=0.5: {predictions}") 
print(f"target: {ground_truth}")
print(f"fraction of misclassified data: {fraction_error:.2f}")
print(f"fraction of misclassified data (with np.mean()): {np.mean(predictions != ground_truth):.2f}")

# Initialize the lists that will contain the errors of each model
nn_train_error = []
nn_cv_error = []

# Build the models
models_bc = utils.build_models()

for model in models_bc:
    
    # Setup the loss and optimizer
    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
    )
    
    print(f"Training {model.name}...")
    
    # Train the model
    model.fit(
        x_bc_train_scaled, y_bc_train,
        epochs=200,
        verbose=0, # 静默模式，不输出训练过程
    )
    
    print("Done!\n")
    
    # Set the threshold for classification
    threshold = 0.5
    
    yhat = model.predict(x_bc_train_scaled)
    yhat = tf.math.sigmoid(yhat) # Apply the sigmoid function to get probabilities
    yhat = np.where(yhat >= threshold, 1, 0) # Apply the threshold
    train_error = np.mean(yhat != y_bc_train) # Compute the error
    nn_train_error.append(train_error) # Save the training error
    
    # Record the fractioin of misclassified data in the cross-validation set
    yhat = model.predict(x_bc_cv_scaled)
    yhat = tf.math.sigmoid(yhat) # Apply the sigmoid function to get probabilities
    yhat = np.where(yhat >= threshold, 1, 0) # Apply the threshold
    cv_error = np.mean(yhat != y_bc_cv) # Compute the error
    nn_cv_error.append(cv_error) # Save the cross-validation error
    
# Print the results
for model_num in range(len(nn_train_error)):
    print(
        f"Model {model_num+1}: Training error: {nn_train_error[model_num]:.5f}, " +
        f"CV error: {nn_cv_error[model_num]:.5f}"
    )
    
    
# Select the model with the lowest error
model_num = 3

# Compute the test error
yhat = models_bc[model_num-1].predict(x_bc_test_scaled)
yhat = tf.math.sigmoid(yhat)
yhat = np.where(yhat >= threshold, 1, 0)
nn_test_error = np.mean(yhat != y_bc_test)

print(f"Selected Model: {model_num}")
print(f"Training Set Classification Error: {nn_train_error[model_num-1]:.4f}")
print(f"CV Set Classification Error: {nn_cv_error[model_num-1]:.4f}")
print(f"Test Set Classification Error: {nn_test_error:.4f}")
    
    
    
    
    






    


























