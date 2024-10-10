# Stock Price Prediction using deep learning and deep reinforcement learning

This project implements a stock price prediction model using deep learning and deep reinforcement learning. The goal is to predict stock prices and evaluate the model's performance using different metrics.

## Model Types
- GRU
- LSTM
- GRU with Dropout
- Bidirectional LSTM
- Bidirectional GRU
- Ensemble

## Evaluation Metrics
- **Mean Squared Error (MSE):** 0.0020
- **Mean Absolute Error (MAE):** 0.0360
- **Root Mean Squared Error (RMSE):** 0.0447
- **Maximum Drawdown (MDD):** 91.8971
- **Return:** 1478.1928
- **Return Of Maximum Drawdown (ROMAD):** 8.0426

## Outputs
The following plots are generated as part of the analysis:

1. Cumulative rewards over epochs
2. Maximum Q-value over epochs
3. Table with headers:
   - Date
   - Close
   - Predicted Close
   - Actual vs. Predicted Close Difference
   - Close Next Day (using heatmap)
4. Actual vs. predicted price plot with buy/sell decisions

Project Description
This project focuses on predicting stock prices using advanced machine learning techniques, specifically neural networks and reinforcement learning. The primary aim is to develop a robust model that can accurately forecast future stock prices based on historical data.

Key Features:

Implementation of various neural network architectures including GRU, LSTM, and their bidirectional counterparts, allowing for flexible experimentation and optimization.
Integration of Q-learning, a reinforcement learning approach, to enhance decision-making processes based on the predicted stock prices.
Calculation of essential evaluation metrics to assess model performance, ensuring that the predictions align closely with actual market behavior.
Achievements:

Developed a model that successfully predicts stock prices with the following performance metrics:
Mean Squared Error (MSE): 0.0020
Mean Absolute Error (MAE): 0.0360
Root Mean Squared Error (RMSE): 0.0447
Maximum Drawdown (MDD): 91.8971
Return: 1478.1928
Return Of Maximum Drawdown (ROMAD): 8.0426
Visual Outputs:

Generated various plots to illustrate the model's performance, including cumulative rewards over epochs, maximum Q-value trends, a heatmap of predicted versus actual values, and a comprehensive visualization of buy/sell decisions based on the model's predictions.
This project serves as a practical demonstration of using machine learning for financial forecasting and provides a solid foundation for further enhancements and research in this domain.