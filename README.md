# Fraud Detection Model Monitoring

This project monitors the performance of a fraud detection machine learning model using NannyML library to detect model drift and performance degradation.

## Overview
Banks use machine learning models to detect fraudulent transactions, but changing data patterns can weaken these defenses. This project helps identify:
- Performance alerts in model accuracy
- Feature drift detection
- Unusual transaction patterns

## Dataset
- **reference.csv**: Historical test data used to establish baseline model performance
- **analysis.csv**: Production data for monitoring model performance over time

### Features
- `timestamp`: Date of the transaction
- `time_since_login_min`: Time since user logged in
- `transaction_amount`: Amount in Pounds (£)
- `transaction_type`: CASH-OUT, PAYMENT, CASH-IN, TRANSFER
- `is_first_transaction`: Binary indicator for first transaction
- `user_tenure_months`: Account age in months
- `is_fraud`: Ground truth fraud label
- `predicted_fraud_proba`: Model prediction probability
- `predicted_fraud`: Model prediction (binary)

## Key Findings
- **Performance Alerts**: [April 2019, May 2019, June 2019]
- **Most Drifted Feature**: time_since_login_min
- **Root Cause**: User login patterns changed significantly, affecting model accuracy

## Installation
```bash
pip install -r requirements.txt
