function [prediction] = test_NN(test_inputs, W1, b1, W2, b2)

    num_test_cases = size(test_inputs, 2);
    h_input = W1' * double(test_inputs) + repmat(b1, 1, num_test_cases);  % Input to hidden layer.
    h_output = 1 ./ (1 + exp(-h_input));  % Output of hidden layer.
    logit = W2' * h_output + repmat(b2, 1, num_test_cases);  % Input to output layer.
    prediction = 1 ./ (1 + exp(-logit));  % Output prediction.