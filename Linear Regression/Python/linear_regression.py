"""
Linear Regression
Created on July 22, 2023
Author: lsEli
GitHub: www.github.com/lsEli
"""

import tensorflow as tf # for machine learning
import numpy as np # for numerical operations
import matplotlib.pyplot as plt # for plotting
import pandas as pd # for data analysis
import plotly.express as px # for plotting
import plotly.graph_objects as go # for plotting

from tensorflow import keras # for machine learning

data_df = pd.DataFrame({ # create a dataframe
    'x': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'y': [1, 2, 3, 4, 5, 6, 7, 8, 9]
})

print("Dataframe:")
print(data_df.head(9)) # print the dataframe

fig = px.scatter(data_df, x='x', y='y') # create a scatter plot
fig.update_traces(marker_size=12) # set the marker size
fig.show() # show the plot

nn_model = keras.Sequential([ # create a neural network model
    keras.layers.Dense(units=1, input_shape=[1]) # add a dense layer
])

nn_model.compile( # compile the model
    optimizer='sgd', # use stochastic gradient descent
    loss='mean_squared_error' # use mean squared error
)

nn_model.set_weights([np.array([[0.]]), np.array([0.])]) # set the weight and bias

print("Model Weights:")
print(nn_model.get_weights()) # show the weights

nn_model.fit( # train the model
    data_df['x'], # input data
    data_df['y'], # output data
    epochs=500 # number of epochs
)

print("Trained Model Weights:")
print(nn_model.get_weights()) # show the weights

predictable_data_df = pd.DataFrame([ # create a dataframe
    [10],
    [11],
    [12],
], columns=['x'])

print("Predictable Dataframe:")
print(predictable_data_df.head(3)) # print the dataframe

predictable_data_df['y'] = nn_model.predict(predictable_data_df['x']) # predict the output

fig = go.Figure() # create a figure

fig.add_trace( # add a trace
    go.Scatter( # create a scatter plot
        x=data_df['x'], # set the x values
        y=data_df['y'], # set the y values
        mode='markers', # set the mode
        name='Historic Data' # set the name
    )
)

fig.add_trace( # add a trace
    go.Scatter( # create a scatter plot
        x=predictable_data_df['x'], # set the x values
        y=predictable_data_df['y'], # set the y values
        mode='lines', # set the mode
        name='Predicted Data', # set the name
        line=go.scatter.Line(color='red') # set the line color
    )
)

fig.update_traces(marker_size=12) # set the marker size

fig.update_layout( # update the layout
    title='Linear Regression', # set the title
    xaxis_title='X', # set the x axis title
    yaxis_title='Y', # set the y axis title
    legend_title='Data' # set the legend title
)

fig.show() # show the plot