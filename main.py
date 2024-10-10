# main section

if __name__ == "__main__":
    model_type = 'gru'  # Change this to 'lstm', 'gru', 'gru_drop', 'bidirectional_lstm', 'bidirectional_gru', or 'ensemble'

    if model_type == 'gru':
        model = create_gru_model()
    elif model_type == 'lstm':
        model = create_lstm_model()
    elif model_type == 'gru_drop':
        model = create_gru_model_with_dropout()
    elif model_type == 'bidirectional_lstm':
        model = create_bidirectional_lstm_model()
    elif model_type == 'bidirectional_gru':
        model = create_bidirectional_gru_model()
    elif model_type == 'ensemble':
        model = create_ensemble_model()
    else:
        raise ValueError(f"Invalid model_type '{model_type}'. Please choose 'lstm', 'gru', 'bidirectional_lstm', 'bidirectional_gru', or 'ensemble'.")

    # Train the model
    history = train_model(model, X_train, y_train, epochs=epochs_DL, batch_size=batch_size_DL)

    # Generate predictions
    predictions = generate_predictions(model, X_test)

    # Calculate evaluation metrics for predictions
    mse, mae, rmse = calculate_metrics(y_test, predictions)

    # Calculate Maximum Drawdown (MDD)
    mdd = calculate_mdd(y_test, predictions)

    # Calculate Simple Return
    initial_value = y_test[0]
    ending_value = y_test[-1]
    simple_return = calculate_simple_return(initial_value, ending_value)

    # Calculate ROMAD
    romad = calculate_romad(simple_return, mdd)

    # Train the Q-learning agent with neural network
    q_agent_nn, cumulative_rewards, max_q_values = train_q_learning_agent_with_nn(predictions, y_test, epochs=epochs_RL)

    # Plot the cumulative rewards over epochs as a line chart
    plot_cumulative_rewards(cumulative_rewards)

    # Plot the maximum Q-value over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(max_q_values) + 1), max_q_values)
    plt.xlabel('Epoch')
    plt.ylabel('Maximum Q-Value')
    plt.title('Maximum Q-Value Over Epochs (Q-Learning with Neural Network)')
    plt.show()

    # Make decisions using the trained Q-learning agent with neural network
    actions_q_learning_nn = make_decision_q_learning_nn(predictions, q_agent_nn)

    # Display the training loss plot
    plot_training_loss(history)

    # Convert dates to a list for plotting
    dates = stock_data['Date'].iloc[-len(y_test):].tolist()

    # Create the table with the requested information
    table_data = create_table(y_test, predictions, actions_q_learning_nn, dates)

    # Plot the table with metrics and Q-values using a heatmap
    plot_table(table_data, epochs=epochs_RL, model_type=model_type, dates=dates)

    # Display the actual vs. predicted price plot with buy/sell decisions
    plot_predictions_with_actions(y_test, predictions, actions_q_learning_nn, dates)