#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer

import urlparse

from random import randint
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Import tensorflow and numpy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

PORT_NUMBER = 8080

COLUMNS = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1","BILL_AMT2","BILL_AMT3",
           "BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","DEFAULT"
          ]

training_data_raw = np.genfromtxt('credit-card-default-training-data.csv',delimiter=',')
holdout_data_raw = np.genfromtxt('credit-card-default-holdout-data.csv',delimiter=',')

print ('Begining tensor flow session')

sess = tf.InteractiveSession()

#Create first 6 elements as input and last element as output into a [6xn] and [1xn] empty NP array

train_data = np.empty((len(training_data_raw) - 2, 24))
holdout_data = np.empty((len(holdout_data_raw) - 2, 24))
train_data_result = np.empty((len(training_data_raw) - 2, 1))
holdout_data_result = np.empty((len(holdout_data_raw) - 2, 1))

# First two rows are just labels so skip them
training_row_index = 0
holdout_row_index = 0
csv_row_index = 0

#Define normalization factors
nf_credit_amount = 250000
nf_payment_amount = 250000
nf_age = 60
nf_generic_factor = 7

#Populate the NP arrays with value from the file.  

for row in training_data_raw:
    if((training_row_index == 0) or (training_row_index == 1)):
        training_row_index = training_row_index + 1
        continue
    j = 0
    #Populate the input. The last element is result, hence [:-1]
    for element in row[:-1]:
        #Normalize if needed
        if j == 0:
            j = j + 1
            continue
        elif j == 1:
            train_data[training_row_index-2][j-1] = float(element)/nf_credit_amount
        elif (j > 11 and j < 24):
            train_data[training_row_index-2][j-1] = float(element)/nf_payment_amount
        elif j == 5:
            train_data[training_row_index-2][j-1] = float(element)/nf_age
        else:
            train_data[training_row_index-2][j-1] = float(element)/nf_generic_factor
        j = j + 1
    #Populate the output    
    train_data_result[training_row_index-2][0] = row[j]
    training_row_index =  training_row_index + 1

for row in holdout_data_raw:
    if((holdout_row_index == 0) or (training_row_index == 1)):
        holdout_row_index = holdout_row_index + 1
        continue
    j = 0
    #Populate the input. The last element is result, hence [:-1]
    for element in row[:-1]:
        #Normalize if needed
        if j == 0:
            j = j + 1
            continue
        elif j == 1:
            holdout_data[holdout_row_index-2][j-1] = float(element)/nf_credit_amount
        elif(j > 11 and j < 24):
            holdout_data[holdout_row_index-2][j-1] = float(element)/nf_payment_amount
        elif j == 5:
            holdout_data[holdout_row_index-2][j-1] = float(element)/nf_age
        else:
            holdout_data[holdout_row_index-2][j-1] = float(element)/nf_generic_factor
        j = j + 1
    #Populate the output    
    holdout_data_result[holdout_row_index-2][0] = row[j]
    holdout_row_index =  holdout_row_index + 1
    
#print('Printing training data')
#print (train_data)

#print('Printing training data results')
#print(train_data_result)

#print('Printing holdout data')
#print (holdout_data)

#print('Printing holdout data results')
#print(holdout_data_result)

# Functions to initialize weights and biases randomly to avoid exploding gradient and dimishing gradient problem

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Start model implementation

# Create the input tensor of [infinite, 6] dimension
x = tf.placeholder(tf.float32, [None, 24])

# Create weights and bias tensor variables for the first layer with 6 inputs and 4 outputs

Wx = weight_variable([24, 20])
bx = bias_variable([20])

# Defind the equation with activation function for the first layer to be RELU

hi = tf.nn.relu(tf.matmul(x, Wx) + bx)

# Create weights and bias tensor variables for the second layer with 4 inputs and 1 output

Whi = weight_variable([20, 10])
bhi = bias_variable([10])

# Defind the equation with activation function for the first layer to be RELU

h1 = tf.nn.relu(tf.matmul(hi, Whi) + bhi)

# Create weights and bias tensor variables for the second layer with 4 inputs and 1 output

Wh1 = weight_variable([10, 1])
bh1 = bias_variable([1])

# Defind the equation with activation function for the first layer to be RELU

h2 = tf.nn.relu(tf.matmul(h1, Wh1) + bh1)

# Create weights and bias tensor variables for the second layer with 4 inputs and 1 output

Wh2 = weight_variable([16, 1])
bh2 = bias_variable([1])

# Defind the activation function for the second layer to simply output a number based on weights and biases

y = tf.nn.relu(tf.matmul(h1, Wh1) + bh1)

# Define loss and optimizer

# Estimated output is an infinite tensor of single dimension
y_ = tf.placeholder(tf.float32, [None, 1])

# Cross entropy is based on L2 loss. [(y-y_)exp2/2*input samples_count]. Reduce index 1 because 0 is just the index of the output tensor
cross_entropy = tf.reduce_sum(tf.pow(y - y_, 2), reduction_indices = [1])/(len(train_data)) #L2 loss

# Learning rate control variable is 0.75. Found through iteration starting from a low value. Big value resulted in missed minima
train_step = tf.train.GradientDescentOptimizer(0.75).minimize(cross_entropy)

# Train
tf.global_variables_initializer().run()

train_index = 0

print('Training start')

# Do batch gradient descent by taking two samples in each step

training_batch_size = 100

for i in range(9000):
    train_data_batch = np.empty((training_batch_size,24))
    train_data_result_batch = np.empty((training_batch_size,1))
    
    #Select random 100 samples and populate them
    for batch_index in range(training_batch_size):
        train_index = randint(0, len(train_data) - 1)
        train_data_result_batch[batch_index][0] = train_data_result[train_index]
        row = train_data[train_index]
        j = 0
        for element in row:
            train_data_batch[batch_index][j] = element
            j = j + 1
    
    #print(train_data_batch)
    #print(train_data_result_batch)
    
    train_step.run({x: train_data_batch, y_: train_data_result_batch})
    
    if i % training_batch_size == 0: 
        print("Iteration =", i)

print('Training complete')

holdout_index = 0

# Set the threshold for the default risk
default_risk_threshold = 0.20
TP = 0
FP = 0
TN = 0
FN = 0

# Calculate TP, TN, FP, FN

risk = [None] * len(holdout_data_result)

for i in range(len(holdout_data)):
    holdout_data_batch = np.empty((1,24))
    
    row = holdout_data[i]
    
    j = 0
    for element in row:
        holdout_data_batch[0][j] = element
        j = j + 1
    
    #print('Holdout data = %s' % holdout_data[i])
    #print('Input = %s' % holdout_data_batch)
    
    get_risk = y
    risk_score = get_risk.eval({x: holdout_data_batch})
    risk[i] = risk_score[0][0]
    
    #print('Predicted Risk = %s - Real Risk = %s' % (risk[i],holdout_data_result[i][0]))
    
    if ((risk[i] > default_risk_threshold) and (holdout_data_result[i][0] == 1)):
        TP = TP + 1
    elif ((risk[i] > default_risk_threshold) and (holdout_data_result[i][0] == 0)):
        FP = FP + 1
    elif ((risk[i] < default_risk_threshold) and (holdout_data_result[i][0] == 1)):
        FN = FN + 1
    elif ((risk[i] < default_risk_threshold) and (holdout_data_result[i][0] == 0)):
        TN = TN + 1
     
        
# lets calculate accuracy to be how many true positives we got right
precision = float(TP)/(TP + FP)
recall = float(TP)/(TP + FN)
accuracy = float(TP)/(TP + FN)
print('Accuracy = %s \nPrecision = %s \nRecall = %s \nTP = %s \nFP = %s \nTN = %s\nFN = %s' % (accuracy, precision, recall, TP, FP, TN, FN))

plot_against_index = 3

if (plot_against_index != -1):
    plot_against_data = [None] * len(holdout_data)
    risk_data = [None] * len(holdout_data_result)
    
    for i in range(len(holdout_data)):
        risk_data[i] = risk[i];
        plot_against_data[i] = holdout_data[i][plot_against_index];
        
    plot_against_data, risk_data = zip(*sorted(zip(plot_against_data, risk_data)))
    
    plot_against_data, risk_data = (list(t) for t in zip(*sorted(zip(plot_against_data, risk_data))))
    
    #plt.figure()
    #plt.plot(plot_against_data, risk_data, 'ro', label='Default Risk')
    #plt.legend()    
    #plt.show()  
    
    xy = np.vstack([plot_against_data,risk_data])
    z = gaussian_kde(xy)(xy)
    
    fig, ax = plt.subplots()
    ax.scatter(plot_against_data, risk_data, c=z, s=100, edgecolor='')
    
    plt.xlabel(COLUMNS[plot_against_index])
    plt.ylabel('Default')
    
    plt.show()
  

  
    
#This class will handles any incoming request from
#the browser 
class myHandler(BaseHTTPRequestHandler):
	
	#Handler for the GET requests
	def do_GET(self):
		self.send_response(200)
		self.send_header('Content-type','text/html')
		self.end_headers()
		# Send the html message
		path = self.path
		params = path.replace('/?', '')
		params_array = params.split('&')
        
		request_data = np.empty((1,6))
        
		for element in params_array:
			if 'x1' in element:
				request_data[0][0] = float(element.replace('x1=',''))
			if 'x2' in element:
				request_data[0][1] = float(element.replace('x2=',''))
			if 'x3' in element:
				request_data[0][2] = float(element.replace('x3=',''))
			if 'x4' in element:
				request_data[0][3] = float(element.replace('x4=',''))
			if 'x5' in element:
				request_data[0][4] = float(element.replace('x5=',''))
			if 'x6' in element:
				request_data[0][5] = float(element.replace('x6=',''))
			if 'x7' in element:
				request_data[0][6] = float(element.replace('x7=',''))
			if 'x8' in element:
				request_data[0][7] = float(element.replace('x8=',''))
			if 'x9' in element:
				request_data[0][8] = float(element.replace('x9=',''))
			if 'x10' in element:
				request_data[0][9] = float(element.replace('x10=',''))
			if 'x11' in element:
				request_data[0][10] = float(element.replace('x11=',''))
			if 'x12' in element:
				request_data[0][11] = float(element.replace('x12=',''))
			if 'x13' in element:
				request_data[0][12] = float(element.replace('x13=',''))
			if 'x14' in element:
				request_data[0][13] = float(element.replace('x14=',''))
			if 'x15' in element:
				request_data[0][14] = float(element.replace('x15=',''))
			if 'x16' in element:
				request_data[0][15] = float(element.replace('x16=',''))
			if 'x17' in element:
				request_data[0][16] = float(element.replace('x17=',''))
			if 'x18' in element:
				request_data[0][17] = float(element.replace('x18=',''))
			if 'x19' in element:
				request_data[0][18] = float(element.replace('x19=',''))
			if 'x20' in element:
				request_data[0][19] = float(element.replace('x20=',''))
			if 'x21' in element:
				request_data[0][20] = float(element.replace('x21=',''))
			if 'x22' in element:
				request_data[0][21] = float(element.replace('x22=',''))
			if 'x23' in element:
				request_data[0][22] = float(element.replace('x23=',''))
			if 'x24' in element:
				request_data[0][23] = float(element.replace('x24=',''))
        
		get_risk = y
		risk_score = get_risk.eval({x: request_data})
		risk = str(risk_score[0][0])
		self.wfile.write(risk)
		return

try:
	#Create a web server and define the handler to manage the
	#incoming request
	server = HTTPServer(('', PORT_NUMBER), myHandler)
	print ('Started httpserver on port ' , PORT_NUMBER)
	
	#Wait forever for incoming htto requests
	#server.serve_forever()

except KeyboardInterrupt:
	print ('^C received, shutting down the web server')
	server.socket.close()

