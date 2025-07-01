from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pickle
import os
import time

class SentimentModel:
    def __init__(self, max_features=10000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features, 
                                          ngram_range=(1, 2),
                                          min_df=5)
        self.model = None
        self.history = None
    
    def build_model(self, model_type="logistic"):
        """
        Build a model for sentiment analysis
        
        Parameters:
        -----------
        model_type : str
            Type of model to build - "logistic" or "random_forest"
        """
        
        if model_type == "logistic":
            self.model = LogisticRegression(C=1.0, 
                                          class_weight='balanced',
                                          solver='liblinear',
                                          random_state=42,
                                          max_iter=200)
        
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100,
                                              max_depth=20,
                                              random_state=42,
                                              class_weight='balanced')
        
        else:
            raise ValueError("model_type must be either 'logistic' or 'random_forest'")
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, **kwargs):
        """
        Train the model
        """
        if self.model is None:
            self.build_model()
        
        # Create a history object to track metrics
        self.history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
        
        # Transform text data to TF-IDF vectors
        start_time = time.time()
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_val_tfidf = self.vectorizer.transform(X_val)
        
        # Train the model
        self.model.fit(X_train_tfidf, y_train)
        
        # Calculate metrics for training data
        y_train_pred = self.model.predict(X_train_tfidf)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Calculate metrics for validation data
        y_val_pred = self.model.predict(X_val_tfidf)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        # Store metrics in history
        self.history['accuracy'] = [train_accuracy]
        self.history['val_accuracy'] = [val_accuracy]
        self.history['loss'] = [1 - train_accuracy]  # Simplified loss
        self.history['val_loss'] = [1 - val_accuracy]  # Simplified loss
        
        end_time = time.time()
        self.training_time = end_time - start_time
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model
        """
        X_test_tfidf = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_tfidf)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Return metrics compatible with the original interface
        return [1 - accuracy, accuracy, precision, recall, f1]
    
    def predict(self, X):
        """
        Make predictions (returns probabilities)
        """
        X_tfidf = self.vectorizer.transform(X)
        
        # For probability outputs
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_tfidf)[:, 1].reshape(-1, 1)
        else:
            # Fallback for models that don't support probabilities
            return self.model.predict(X_tfidf).reshape(-1, 1)
    
    def save_model(self, filepath='sentiment_model.pkl'):
        """
        Save the model to disk
        """
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'vectorizer': self.vectorizer}, f)
    
    def load_model(self, filepath='sentiment_model.pkl'):
        """
        Load a model from disk
        """
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                saved = pickle.load(f)
                self.model = saved['model']
                self.vectorizer = saved['vectorizer']
            return True
        return False

def explain_activation_functions():
    """
    Return explanation of activation functions used in the model
    """
    explanations = {
        'ReLU (Rectified Linear Unit)': {
            'formula': 'f(x) = max(0, x)',
            'description': 'ReLU is a popular activation function that outputs the input directly if it is positive, otherwise, it outputs zero. It helps models learn faster and mitigates the vanishing gradient problem.',
            'advantages': ['Computationally efficient', 'Helps with sparse activation', 'Reduces likelihood of vanishing gradient problem'],
            'disadvantages': ['Can lead to "dying ReLU" problem where neurons can get stuck', 'Not zero-centered']
        },
        'Sigmoid': {
            'formula': 'f(x) = 1 / (1 + e^-x)',
            'description': 'Sigmoid transforms input values to between 0 and 1, making it suitable for binary classification output layers.',
            'advantages': ['Outputs probability between 0 and 1', 'Smooth gradient', 'Clear predictions'],
            'disadvantages': ['Suffers from vanishing gradient problem', 'Not zero-centered', 'Computationally expensive']
        },
        'Tanh (Hyperbolic Tangent)': {
            'formula': 'f(x) = (e^x - e^-x) / (e^x + e^-x)',
            'description': 'Tanh squashes input values to between -1 and 1, making it zero-centered.',
            'advantages': ['Zero-centered', 'Stronger gradients than sigmoid'],
            'disadvantages': ['Still suffers from vanishing gradient problem', 'Computationally expensive']
        }
    }
    
    return explanations

def explain_cross_entropy():
    """
    Return explanation of cross-entropy loss
    """
    explanation = {
        'formula': 'L = -Σ[y * log(p) + (1-y) * log(1-p)]',
        'description': 'Cross-entropy loss measures the performance of a classification model whose output is a probability value between 0 and 1. It increases as the predicted probability diverges from the actual label.',
        'binary_case': 'For binary classification, we use binary cross-entropy: -(y * log(p) + (1-y) * log(1-p))',
        'why_used': 'It penalizes confident wrong predictions more than less confident ones, making it ideal for classification tasks.'
    }
    
    return explanation
