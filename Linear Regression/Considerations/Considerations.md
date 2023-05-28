# Considerations

## Pandas Dataframe

A pandas DataFrame is a two-dimensional, size-mutable, and heterogeneous data structure provided by the pandas library in Python. It is one of the most widely used data structures for data analysis and manipulation in the field of data science.

In simple terms, a DataFrame is like a table, where data is organized in rows and columns. Each column in the DataFrame represents a variable, while each row represents an individual observation or data point. This tabular structure makes it easy to work with and analyze data in a way that resembles working with spreadsheet-like data.

Key characteristics of a pandas DataFrame:

- **Two-dimensional**: Data is arranged in rows and columns, forming a grid-like structure.
- **Size-mutable**: You can add or remove rows and columns after the DataFrame's creation.
- **Heterogeneous data types**: Each column can contain different types of data (e.g., numbers, strings, dates).
- **Labeled axes**: Both rows and columns have associated labels, making it easier to access and manipulate data.
- **Powerful data manipulation tools**: The pandas library provides a wide range of functions and methods to perform data filtering, aggregation, grouping, merging, and more.

## Dataframe head

In pandas, `dataframe.head()` method is used to display the first few rows of a DataFrame. It is a convenient way to get a quick overview of the data contained in the DataFrame without displaying the entire contents, especially when dealing with large datasets.

The `dataframe.head()` method returns the first 5 rows by default, but you can specify the number of rows you want to display by passing an integer argument to the method. For example, `dataframe.head(10)` will display the first 10 rows of the DataFrame.

## Keras Sequential

The `keras.Sequential` class is used to create a linear stack of layers in a neural network. It allows you to build a neural network model by adding one layer at a time. Each layer added to the Sequential model represents a specific type of neural network layer, such as a fully connected (dense) layer, a convolutional layer, a recurrent layer, etc.

## Keras Layers Dense

In Keras, the `keras.layers.Dense` class represents a fully connected (dense) layer in a neural network. It is one of the fundamental building blocks for constructing deep learning models.

A **dense layer** consists of nodes, also called neurons or units, where each node is connected to every node in the previous layer. The connections between nodes have associated weights and biases, which are learned during the training process. These weights and biases allow the dense layer to perform transformations on the input data and learn to represent complex patterns in the data.

## Unit

An **Unit**,  refers to a node or neuron in a neural network layer. Each unit takes input from the previous layer (or directly from the input data in the case of the first layer) and performs a computation to produce an output. The output from a unit is then passed as input to the units in the next layer.

In a dense (fully connected) layer, each unit is connected to every unit in the previous layer. Each connection has an associated weight and a bias, which are learned during the training process. These weights and biases allow the neural network to learn to represent patterns and relationships in the data.

The computation performed by a unit is a linear combination of its inputs, followed by an activation function. The activation function introduces non-linearity into the network, allowing it to learn complex relationships in the data.

For example, in a dense layer with 32 units, each unit takes the input from the previous layer, applies a linear combination (sum of weighted inputs and biases), and then passes the result through the activation function. This process is repeated for all 32 units in the layer.

The number of units in a layer is a hyperparameter that can be adjusted during the model's architecture design. The number of units affects the model's capacity to learn complex patterns from the data. Too few units may result in underfitting (the model cannot learn the data's underlying patterns), while too many units may lead to overfitting (the model memorizes the training data and fails to generalize to new data).

Choosing the appropriate number of units and designing the neural network architecture is a crucial step in building an effective deep learning model. This decision is often based on empirical experimentation and best practices for the specific task at hand.

## Activation function

An **Activation function**, refers to a mathematical function applied to the output of a neuron or unit in a neural network layer. The activation function introduces non-linearity into the network, which is essential for the network to learn and approximate complex relationships in the data.

The activation function takes the weighted sum of the inputs to a neuron, adds a bias term, and then applies the function to produce the output of the neuron. This output is then passed as input to the neurons in the next layer.

Without activation functions, the neural network would be reduced to a series of linear transformations, and the entire network would behave like a single linear function. Adding non-linearity through activation functions allows the network to learn more sophisticated and intricate patterns in the data, making it capable of solving a wide range of complex problems.

### Common activation functions used in neural networks

- **Sigmoid**: The sigmoid activation function maps the input to a value between 0 and 1. It is often used in the output layer of binary classification problems.

- **ReLU (Rectified Linear Unit)**: ReLU activation sets negative values to zero and keeps non-negative values unchanged. It is widely used in hidden layers due to its simplicity and ability to mitigate the vanishing gradient problem.

- **Leaky ReLU**: Similar to ReLU, but it allows a small negative slope for negative inputs, which can help with the "dying ReLU" problem where neurons may become inactive during training.

- **Tanh (Hyperbolic Tangent)**: The tanh activation function maps the input to a value between -1 and 1. It is often used in hidden layers as an alternative to ReLU.

- **Softmax**: The softmax activation function is used in the output layer for multi-class classification tasks. It converts raw scores into a probability distribution, allowing the model to predict the class with the highest probability.

The choice of activation function is an important hyperparameter that affects the performance and behavior of the neural network. Different activation functions may be more suitable for different types of problems and architectures, and selecting the right activation function can significantly impact the success of the neural network in learning from the data.

## Input Shape

**Input Shape**, refers to the shape or dimensions of the input data that is fed into the neural network for training or making predictions. The input shape is a crucial parameter that needs to be specified correctly when defining the architecture of a neural network.

For example, in a fully connected (dense) neural network, the input shape is determined by the number of features or variables in the input data. If you have a dataset with 100 samples and each sample has 10 features, the input shape would be (100, 10). The first dimension represents the number of samples in the dataset, and the second dimension represents the number of features in each sample.

When creating a neural network model using deep learning libraries like Keras or TensorFlow, you typically need to specify the input shape of the first layer. Subsequent layers will automatically infer their input shape from the previous layer's output shape.

## Optimizers

An **Optimizer**, refers to an algorithm or method used to update the model's weights during the training process. The main goal of the optimizer is to minimize the loss function, which measures the difference between the model's predictions and the actual target values (labels) on the training data.

The optimization process is an essential part of training a neural network, as it determines how the model learns from the data and adjusts its weights to improve its performance on the given task. Different optimizers use various strategies to find the optimal set of weights that result in the best model performance.

### Common optimizers used in deep learning

- **Stochastic Gradient Descent (SGD)**: The basic form of gradient descent, where the weights are updated after processing each individual data point (batch size = 1). This approach can be noisy and slow to converge but is straightforward and easy to implement.

- **Mini-Batch Gradient Descent**: An improvement over SGD, where the weights are updated after processing a small batch of data points (batch size > 1). Mini-batch gradient descent strikes a balance between efficiency and noise in the weight updates, leading to faster convergence.

- **Adam (Adaptive Moment Estimation)**: A popular adaptive learning rate optimization algorithm that combines ideas from RMSprop and momentum. Adam adapts the learning rates of each weight parameter based on past gradients, making it well-suited for a wide range of deep learning tasks.

- **RMSprop (Root Mean Square Propagation)**: An optimization algorithm that adjusts the learning rates of individual parameters based on the average of recent squared gradients. It helps overcome the problem of rapidly decreasing learning rates in traditional gradient descent.

- **Adagrad (Adaptive Gradient Algorithm)**: An adaptive learning rate optimization algorithm that adapts the learning rate of each weight parameter based on the historical gradients. It tends to give larger updates for infrequent parameters and smaller updates for frequent ones.

- **Adadelta**: An extension of Adagrad that addresses some of its limitations by using a moving window of gradient updates instead of storing all past gradients.

- **Nadam**: An extension of Adam that incorporates Nesterov accelerated gradient (NAG) method, which typically leads to faster convergence.

## Loss

**Loss**, refers to a measure of how well the model's predictions match the actual target values (labels) on the training data. It quantifies the difference between the predicted output and the ground truth for each data point in the training set.

The loss function, also known as the cost function or objective function, is a critical component of the training process. The goal of training a neural network is to minimize the value of the loss function, which means making the model's predictions as close as possible to the actual target values.

Different types of problems require different loss functions. For example:

- **Binary Classification**: In binary classification problems, where there are only two possible classes (e.g., "yes" or "no," "spam" or "not spam"), the most common loss function used is binary cross-entropy (also known as log loss). It measures the difference between the true labels (0 or 1) and the predicted probabilities of the positive class (ranging from 0 to 1).

- **Multi-Class Classification**: For multi-class classification tasks, where there are more than two classes (e.g., recognizing different types of animals or digits), the categorical cross-entropy loss function is commonly used. It calculates the difference between the true one-hot encoded labels and the predicted class probabilities.

- **Regression**: In regression problems, where the task is to predict a continuous value (e.g., predicting house prices, temperature), mean squared error (MSE) is a common loss function. It measures the average squared difference between the true values and the predicted values.

During the training process, the model's weights are adjusted based on the gradients of the loss function with respect to the model's parameters. The optimization algorithm (e.g., SGD, Adam) uses these gradients to update the weights in a way that minimizes the loss function and improves the model's performance on the training data.

Choosing an appropriate loss function is crucial because it directly influences how the model learns from the data and what it optimizes during training. Selecting the right loss function depends on the specific problem you are trying to solve and the nature of your target variable.

## Epochs

The **Epochs**, refers to the number of times the entire dataset is passed forward and backward through the neural network during the training process. Each pass through the entire dataset is called one epoch.

During training, the neural network adjusts its weights and biases based on the input data and the corresponding target values (labels) to minimize the difference between the predicted outputs and the actual outputs. The training process is typically done through an optimization algorithm such as **Stochastic Gradient Descent (SGD)** or its variants.

One epoch consists of multiple iterations, where each iteration processes one batch of data. The batch size is the number of samples processed in one iteration. For example, if you have 1,000 training samples and a batch size of 32, each epoch will consist of 1,000 / 32 = 31.25 iterations (usually rounded down to 31).

## Batch size

A **Batch size** refers to a subset of the training data that is used together to update the model's weights during one iteration of the training process. Instead of updating the model's weights after each individual data point, batching allows for more efficient training by updating the weights after processing multiple data points at once.

The process of training a neural network involves computing the model's predictions for the input data, calculating the prediction errors (also known as "loss" or "cost"), and then using an optimization algorithm (e.g., Stochastic Gradient Descent - SGD) to adjust the model's weights to minimize the loss.

In stochastic gradient descent, a batch size of 1 is used, meaning that the model's weights are updated after processing each individual data point. This approach can lead to noisy weight updates and slower convergence since the updates are based on a single data point.

Using larger batch sizes, such as 32, 64, or 128, is a common practice. With larger batches, the optimization algorithm updates the weights based on the average loss over the entire batch, which provides a more stable and accurate estimate of the direction in which the weights should be adjusted. This can lead to faster convergence and more efficient use of computational resources.

The size of the batch is a hyperparameter that needs to be chosen based on factors such as the available memory, the size of the dataset, and the computational resources available. Smaller batch sizes introduce more noise into the weight updates but may allow for faster convergence in some cases. On the other hand, larger batch sizes can lead to smoother weight updates but require more memory to store the intermediate computations.

In **Keras** or other deep learning libraries, the batch size can be specified when training the model using the `fit()` method.
