import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import pandas as pd
from wordcloud import WordCloud
from collections import Counter

def plot_training_history(history):
    """
    Plot training and validation accuracy and loss
    For scikit-learn models, history is a dictionary with metrics
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    # For scikit-learn models, we expect a dictionary
    # If there are metrics for multiple epochs, we can plot them
    # Otherwise, we'll just show the single values as horizontal lines
    
    # Check if we have a TensorFlow history object or our custom dictionary
    if isinstance(history, dict):
        # Our scikit-learn dictionary with single values
        # Accuracy plot
        ax[0].axhline(y=history['accuracy'][0], color='blue', linestyle='-', label='Training Accuracy')
        ax[0].axhline(y=history['val_accuracy'][0], color='orange', linestyle='-', label='Validation Accuracy')
        
        # Loss plot
        ax[1].axhline(y=history['loss'][0], color='blue', linestyle='-', label='Training Loss')
        ax[1].axhline(y=history['val_loss'][0], color='orange', linestyle='-', label='Validation Loss')
    else:
        # TensorFlow history object
        # Accuracy plot
        ax[0].plot(history.history['accuracy'], label='Training Accuracy')
        ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        
        # Loss plot
        ax[1].plot(history.history['loss'], label='Training Loss')
        ax[1].plot(history.history['val_loss'], label='Validation Loss')
    
    # Set titles and labels
    ax[0].set_title('Model Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    ax[0].grid(True)
    
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    ax[1].grid(True)
    
    plt.tight_layout()
    
    return fig

def plot_confusion_matrix(y_true, y_pred):
    """
    Plot confusion matrix using Plotly
    """
    y_pred_binary = (y_pred > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # Calculate percentages for annotations
    total = cm.sum()
    percentage_cm = cm / total * 100
    
    # Create labels for the confusion matrix
    labels = ["Negative", "Positive"]
    
    # Create the heatmap
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        color_continuous_scale='Blues',
        labels=dict(x="Predicted", y="Actual", color="Count"),
        title="Confusion Matrix"
    )
    
    # Add annotations with counts and percentages
    annotations = []
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=f"{cm[i, j]}<br>({percentage_cm[i, j]:.1f}%)",
                    showarrow=False,
                    font=dict(color="white" if cm[i, j] > cm.max() / 2 else "black")
                )
            )
    
    fig.update_layout(annotations=annotations)
    
    return fig

def plot_roc_curve(y_true, y_pred):
    """
    Plot ROC curve using Plotly
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr, 
            y=tpr, 
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})'
        )
    )
    
    # Add diagonal reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1], 
            y=[0, 1], 
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Random Classifier'
        )
    )
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate', scaleanchor="x", scaleratio=1),
        showlegend=True
    )
    
    return fig

def plot_precision_recall_curve(y_true, y_pred):
    """
    Plot Precision-Recall curve using Plotly
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    
    fig = go.Figure()
    
    # Add PR curve
    fig.add_trace(
        go.Scatter(
            x=recall, 
            y=precision, 
            mode='lines',
            name='Precision-Recall Curve'
        )
    )
    
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis=dict(title='Recall'),
        yaxis=dict(title='Precision'),
        showlegend=True
    )
    
    return fig

def plot_word_distribution(df, column, sentiment):
    """
    Create a word cloud from the text data based on sentiment
    """
    # Filter the dataframe by sentiment
    if sentiment == 'positive':
        filtered_df = df[df['sentiment'] == 'positive']
    else:
        filtered_df = df[df['sentiment'] == 'negative']
    
    # Combine all text
    all_text = ' '.join(filtered_df[column].tolist())
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=100,
        contour_width=3
    ).generate(all_text)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Word Cloud for {sentiment.capitalize()} Reviews')
    
    return fig

def plot_sentiment_distribution(df):
    """
    Plot the distribution of sentiment in the dataset
    """
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    fig = px.bar(
        sentiment_counts, 
        x='Sentiment', 
        y='Count', 
        color='Sentiment',
        color_discrete_map={'positive': 'green', 'negative': 'red'},
        title='Distribution of Sentiments in Dataset'
    )
    
    return fig

def plot_review_length_distribution(df):
    """
    Plot the distribution of review lengths
    """
    df['review_length'] = df['review'].apply(len)
    
    fig = px.histogram(
        df, 
        x='review_length', 
        color='sentiment',
        color_discrete_map={'positive': 'green', 'negative': 'red'},
        nbins=50,
        title='Distribution of Review Lengths',
        labels={'review_length': 'Review Length (characters)', 'count': 'Count'}
    )
    
    return fig

def plot_activation_functions():
    """
    Plot common activation functions
    """
    x = np.linspace(-5, 5, 100)
    
    # Calculate activation function values
    relu = np.maximum(0, x)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=x, y=relu, mode='lines', name='ReLU'))
    fig.add_trace(go.Scatter(x=x, y=sigmoid, mode='lines', name='Sigmoid'))
    fig.add_trace(go.Scatter(x=x, y=tanh, mode='lines', name='Tanh'))
    
    fig.update_layout(
        title='Common Activation Functions',
        xaxis=dict(title='x'),
        yaxis=dict(title='f(x)'),
        legend=dict(x=0.01, y=0.99),
        hovermode='closest'
    )
    
    return fig

def plot_binary_cross_entropy():
    """
    Plot binary cross-entropy loss function
    """
    y_pred = np.linspace(0.001, 0.999, 100)
    
    # Calculate loss for true label y=1
    loss_y1 = -np.log(y_pred)
    
    # Calculate loss for true label y=0
    loss_y0 = -np.log(1 - y_pred)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=y_pred, y=loss_y1, mode='lines', name='True Label y=1'))
    fig.add_trace(go.Scatter(x=y_pred, y=loss_y0, mode='lines', name='True Label y=0'))
    
    fig.update_layout(
        title='Binary Cross-Entropy Loss',
        xaxis=dict(title='Predicted Probability p'),
        yaxis=dict(title='Loss Value -log(p) or -log(1-p)'),
        legend=dict(x=0.01, y=0.99),
        hovermode='closest'
    )
    
    return fig
