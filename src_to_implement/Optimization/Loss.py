import numpy as np

class CrossEntropyLoss:
    def __init__(self) -> None:
        # Constructor method, no specific initialization required
        pass

    def forward(self, predicted_values, true_labels) -> float:
        """
        Forward pass method for Cross Entropy Loss calculation.

        Parameters:
        - predicted_values: The predicted values from the model (output tensor).
        - true_labels: The true labels for the corresponding input samples.

        Returns:
        - loss: Calculated Cross Entropy Loss.
        """
        # Storing predicted values for later use in the backward pass
        self.predicted_values = predicted_values
        
        # Calculating the Cross Entropy Loss
        # Adding a small epsilon to avoid numerical instability (log(0))
        loss = np.sum(-np.log(predicted_values[true_labels == 1] + np.finfo(float).eps))
        return loss

    def backward(self, true_labels) -> np.ndarray:
        """
        Backward pass method for Cross Entropy Loss gradient calculation.

        Parameters:
        - true_labels: The true labels for the corresponding input samples.

        Returns:
        - gradient: Calculated gradient of the Cross Entropy Loss.
        """
        # Calculating the gradient of the Cross Entropy Loss
        # Adding a small epsilon to avoid division by zero
        gradient = -np.divide(true_labels, self.predicted_values + np.finfo(float).eps)
        return gradient
