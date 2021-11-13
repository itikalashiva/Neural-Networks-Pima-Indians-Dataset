# Neural-Networks-Pima-Indians-Dataset

Neural networks, also known as artificial neural networks (ANNs) or simulated neural networks (SNNs),

are a subset of machine learning that provide the foundation of deep learning techniques. Their name and

form are inspired by the human brain, and they imitate the way real neurons communicate with one

another.

Neural Network is a layered architecture, it contains an input layer, optional number of hidden layers and

an output layer.

![alt text](https://github.com/itikalashiva/Nueral-Networks-Pima-Indians-Dataset/blob/main/Screenshots/NN.PNG)

The above figure is a visual representation of the Neural network for binary classification, where n

number of neurons map to a single value in the output.

<hr>

**8.2 Computation of Neural networks**

Computation of Neural Network is done through forward propagation and computation of gradients is

done by backward propagation

<hr>

**8.2.1 Forward Propagation**

Forward Propagation is a process of calculation and storage of intermediate values from input to output

layers. The value of each neuron in the hidden layer is represented as an activation function applied on

Z (weighted sum of inputs).

**Y= Activation Function(Z)**

**8.2.2 Activation Functions**

Activation functions help in transforming the linear inputs to nonlinear outputs.

There are many activation functions like sigmoid, Relu, Tanh, softmax. Relu has been used in

this project.

<hr>

**ReLU (Rectified Linear Unit)**

Rectified Linear activation function is a piece wise linear function that will output the input directly if it is

positive, otherwise it will output zero. The function solves the problem of vanishing gradients,

letting models to train quicker and perform better.


**8.2.3 Loss function**

As the given project is a binary classification problem (The output values would be either 0 or 1)

, Binary Cross Entropy can be used as a loss function.

Value of every neuron in the second layer is computed as the weighted sum of the inputs from

the input layer.

**Back Propagation**

Backpropagation is a method used to train specific types of neural networks - it is simply a

principle that allows the machine learning software to adapt itself based on its previous function.

**8.2.4 Optimizer**

**Adam**

Adam is characterized as "a technique for efficient stochastic optimization using just first-order

gradients and requiring little memory."

<hr>

**8.3 Implementation**

Neural Networks is implemented using keras and Neural Networks. Keras an open-source library

is used for the implementation of this Artificial Neural Network Model.

The Sequential model API is a method of creating learning models in which a Sequential class

representation is established, and model layers are built and added to it. The dense layer is a

neural network layer that is highly linked, which implies that each neuron in the dense layer

receives input from all neurons in the preceding layer.

Compilation is the final stage in generating a model before training. While compiling, various

parameters such as optimizers, loss, and metrics must be considered.

As the parameters are not tuned the accuracy is very low.

**Hyper parameter tuning**

List of Hyper parameters

\1. Number of Neurons

\2. Learning Rate

\3. Penalty (Lambda)

\4. Epochs (No. of iterations)

\5. Batch Size


**Results:**

**Test Accuracy**

By passing the test data set values to the above trained model, it is giving an accuracy of 80.52%.

![alt text](https://github.com/itikalashiva/Nueral-Networks-Pima-Indians-Dataset/blob/main/Screenshots/accuracy.PNG)
![alt text](https://github.com/itikalashiva/Nueral-Networks-Pima-Indians-Dataset/blob/main/Screenshots/loss.PNG)

**Conclusion**

We can confidently state that the model is neither overfitting nor underfitting because the

validation loss was lower than the training loss. Furthermore, on unseen data from the testing set,

the model had an accuracy of 80% in both Logistic Regression and Neural Networks. As a result,

if the model is used in the real world, it can be confidently stated that it will perform well.
