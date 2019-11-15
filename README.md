# gradient_based_learning
As part of Deep Learning for Text and Sequences course I implemented a log-linear classifier and a multi-layer
perceptron classifier.I also write the training code for them.
All the classifiers uses the hard cross-entropy loss.
I tested the classifiers on the language identification task, and the XOR task.

1. XOR task - given a 2D input vector, return the vector's xor.
2. language identification task - given a text written in some language, return in what language the text is written

# Gradient-Based Learning Project:  
1. [Introduction](#introduction)
1. [Gradient Checking](#introduction)
2. [Dependencies:](#dependencies) 
3. [Installation:](#installation)


## Introduction
As part of Deep Learning for Text and Sequences course I implemented a log-linear classifier and a multi-layer
perceptron classifier.I also write the training code for them.
All the classifiers uses the hard cross-entropy loss.
I tested the classifiers on the language identification task, and the XOR task.

## Gradient Checking
As part of the project I needed to calculate and implement the "Backpropagation" of the model.
For doing so, I created the gradient check which compare between
the calculated gradient approximation of:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;f'(x)\approx\frac{f(x+\epsilon)-f(x-\epsilon)}{2\epsilon}" title="\Large f'(x)\approx\frac{f(x+\epsilon)-f(x-\epsilon)}{2\epsilon}" />

## Dependencies:
* Windows / Linux
* Git

## Installation:
1. Clone the repository:  
    ```
    $ git clone https://github.com/eyalcohen308/gradient_based_learning
    ```
2. run this commands:
    ```
    $ make compile
    $ make run
    ```
