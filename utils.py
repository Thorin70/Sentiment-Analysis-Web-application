import pandas as pd
import numpy as np
import time
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def timer_decorator(func):
    """
    Decorator to time function execution
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} took {execution_time:.2f} seconds to execute.")
        return result, execution_time
    return wrapper

def get_example_reviews():
    """
    Return a list of example reviews for interactive testing
    """
    positive_examples = [
        "This movie is absolutely amazing! The acting was superb and the plot kept me engaged throughout.",
        "One of the best films I've seen this year. Brilliant performances by the entire cast.",
        "A wonderful little production with excellent performances and a heartwarming story.",
        "I thoroughly enjoyed this film. The cinematography was beautiful and the story was compelling.",
        "What an incredible film! The director has created a masterpiece that will be remembered for years."
    ]
    
    negative_examples = [
        "This is possibly the worst movie I've ever seen. The plot made no sense and the acting was terrible.",
        "A complete waste of time. Don't bother watching this movie.",
        "I was really disappointed with this film. The storyline was predictable and the dialogue was poor.",
        "Awful movie with terrible acting and a completely nonsensical plot.",
        "I couldn't even finish watching this film. It was boring and poorly executed."
    ]
    
    return positive_examples, negative_examples

def create_sample_neural_network():
    """
    Create a simple neural network visualization for educational purposes
    """
    
    from sklearn.neural_network import MLPClassifier
    
    model = MLPClassifier(
        hidden_layer_sizes=(3, 2),
        activation='relu',
        solver='adam',
        max_iter=200,
        random_state=42
    )
    
    # This is just for visualization, not actual training
    return model

def display_cross_entropy_calculation():
    """
    Display step-by-step calculation of cross-entropy loss for educational purposes
    """
    # Example predictions and true labels
    y_true = [1, 0, 1, 0]
    y_pred = [0.9, 0.2, 0.8, 0.3]
    
    # Calculate loss for each sample
    losses = []
    for yt, yp in zip(y_true, y_pred):
        if yt == 1:
            loss = -np.log(yp)
        else:
            loss = -np.log(1 - yp)
        losses.append(loss)
    
    # Create DataFrame for display
    df = pd.DataFrame({
        'True Label': y_true,
        'Predicted Probability': y_pred,
        'Log Loss Formula': [f"-log({p})" if t == 1 else f"-log(1-{p})" for t, p in zip(y_true, y_pred)],
        'Loss Value': losses
    })
    
    # Calculate mean loss
    mean_loss = np.mean(losses)
    
    return df, mean_loss

def compute_class_weights(y_train):
    """
    Compute class weights for imbalanced datasets
    """
    # Count class occurrences
    unique_classes = np.unique(y_train)
    class_counts = np.bincount(y_train.astype(int))
    
    # Calculate weights
    total_samples = len(y_train)
    n_classes = len(unique_classes)
    
    weights = {}
    for i, cls in enumerate(unique_classes):
        weights[cls] = total_samples / (n_classes * class_counts[int(cls)])
    
    return weights

def explain_neural_network_components():
    """
    Return explanations of key neural network components
    """
    explanations = {
        'Neuron': 'The basic computational unit of a neural network that receives inputs, applies weights and biases, and passes the result through an activation function.',
        
        'Weights': 'Learnable parameters that determine the strength of connection between neurons. They are adjusted during training to minimize the loss function.',
        
        'Biases': 'Additional learnable parameters added to the weighted sum before applying the activation function. They allow the neuron to shift the activation function left or right.',
        
        'Activation Function': 'A mathematical function applied to the output of a neuron to introduce non-linearity, enabling the network to learn complex patterns.',
        
        'Layer': 'A group of neurons that process a set of inputs and produce a set of outputs. Different types include input, hidden, and output layers.',
        
        'Feedforward': 'The process where input signals propagate through the network from input to output without loops or cycles.',
        
        'Backpropagation': 'The algorithm used to calculate gradients for each weight in the network by propagating the error backward from the output.',
        
        'Gradient Descent': 'An optimization algorithm that iteratively adjusts weights to minimize the loss function by moving in the direction of the steepest descent of the gradient.',
        
        'Epoch': 'One complete pass through the entire training dataset during the training process.',
        
        'Batch': 'A subset of the training data used in one iteration of model training.',
        
        'Learning Rate': 'A hyperparameter that controls how much the model weights are adjusted with respect to the loss gradient.'
    }
    
    return explanations
