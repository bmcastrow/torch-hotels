# Hotel Reservation/Cancellation Classification

This project focuses on classifying (if we call it "predicting" it's not wrong) hotel booking cancellations and reservation statuses leveraging the power of ML/"DL" with Pytorch. Two primary tasks are addressed:

1. **Binary Classification:** Predict whether a customer will cancel their hotel booking (`is_canceled` column).
2. **Multiclass Classification:** Classify the reservation status (`reservation_status` column) into three categories:
   - `Check-Out` (label: 2)
   - `Canceled` (label: 1)
   - `No-Show` (label: 0)

## Dataset

The dataset used in this project is from Kaggle's "Hotel Booking Demand" dataset. You can download it [here](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand?resource=download).

Key dataset details:
- Binary target column: `is_canceled`
- Multiclass target column: `reservation_status`
- Features include booking details, customer information, stay information...

## Model Architecture

**The architectures are intentionally simple due to limited computational resources. The primary objective was to gain hands-on experience with cutting-edge ML and DL frameworks while deepening understanding of key concepts in the field.**

### Binary Classification Model
A simple feedforward neural network:
- Input layer: 73 features
- Hidden layers:
  - Layer 1: 64 nodes, ReLU activation
  - Layer 2: 32 nodes, ReLU activation
- Output layer: 1 node, Sigmoid activation
- Loss function: Binary Cross-Entropy Loss
- Optimizer: Adam

### Multiclass Classification Model
Another feedforward neural network:
- Input layer: 73 features
- Hidden layers:
  - Layer 1: 73 nodes, ReLU activation
  - Layer 2: 36 nodes, ReLU activation
- Output layer: 3 nodes, Softmax activation
- Loss function: Cross-Entropy Loss
- Optimizer: Adam

## Future Improvements

- Collect more data to address class imbalance.
- Optimize the neural network architecture (e.g., more layers, regularization, advanced optimizers).
- Fine-tune hyperparameters (learning rate, batch size, etc.).
- Improve feature engineering to include more meaningful predictors.

## How to Run

1. Clone this repository.
2. Download the dataset from Kaggle and place it in the root directory.
3. Run the `hotelsclassification.py` script using Python.

```bash
python hotelsclassification.py
```

# Requirements

- Computer
- Python 3.x
```bash
pip install -r requirements.txt


