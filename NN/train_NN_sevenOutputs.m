
function [train_CE_list, valid_CE_list, W1, b1, W2, b2] = train_NN_sevenOutputs(train_inputs, ... 
    valid_inputs, train_labels, valid_labels, num_hiddens, num_epochs, ...
    momentum, eps)
   %% initialize
    num_inputs = size(train_inputs, 1);
    num_outputs = 7;
   
    %%% make random initial weights smaller, and include bias weights
    W1 = 0.01 * randn(num_inputs, num_hiddens);
    b1 = zeros(num_hiddens, 1);
    W2 = 0.01 * randn(num_hiddens, num_outputs);
    b2 = zeros(num_outputs, 1);

    dW1 = zeros(size(W1));
    dW2 = zeros(size(W2));
    db1 = zeros(size(b1));
    db2 = zeros(size(b2));
    
    total_epochs = 0; % number of learning epochs so far. This is incremented by numEpochs each time runbp is called.
    %% Training
    % (1, num_epochs): cross entropy, (2, num_epochs) -> classification
    % error
    train_CE_list = zeros(2, num_epochs);
    valid_CE_list = zeros(2, num_epochs);

    num_train_cases = size(train_inputs, 2);
    num_valid_cases = size(valid_inputs, 2);
    %%
    for epoch = 1:num_epochs
      % Fprop
      h_input = W1' * double(train_inputs) + repmat(b1, 1, num_train_cases);  % Input to hidden layer.
      h_output = 1 ./ (1 + exp(-h_input));  % Output of hidden layer.
      logit = W2' * h_output + repmat(b2, 1, num_train_cases);  % Input to output layer.
      prediction = 1 ./ (1 + exp(-logit));  % Output prediction.

      % Compute cross entropy
      train_CE = -mean(mean(sum(train_labels .* log(prediction') + ...
          (1 - train_labels) .* log(1 - prediction'), 2)));
      train_classErr = 1 - sum(sum(train_labels .* log_to_binary(prediction'), 2)) / double(num_train_cases);
      % Compute deriv
      dEbydlogit = prediction - train_labels';

      % Backprop
      dEbydh_output = W2 * dEbydlogit;
      dEbydh_input = dEbydh_output .* h_output .* (1 - h_output) ;

      % Gradients for weights and biases.
      dEbydW2 = h_output * dEbydlogit';
      dEbydb2 = sum(dEbydlogit, 2);
      dEbydW1 = double(train_inputs) * dEbydh_input';
      dEbydb1 = sum(dEbydh_input, 2);

      %%%%% Update the weights at the end of the epoch %%%%%%
      dW1 = momentum * dW1 - (eps / num_train_cases) * dEbydW1;
      dW2 = momentum * dW2 - (eps / num_train_cases) * dEbydW2;
      db1 = momentum * db1 - (eps / num_train_cases) * dEbydb1;
      db2 = momentum * db2 - (eps / num_train_cases) * dEbydb2;

      W1 = W1 + dW1;
      W2 = W2 + dW2;
      b1 = b1 + db1;
      b2 = b2 + db2;

      %%%%% Test network's performance on the valid patterns %%%%%
      h_input = W1' * double(valid_inputs) + repmat(b1, 1, num_valid_cases);  % Input to hidden layer.
      h_output = 1 ./ (1 + exp(-h_input));  % Output of hidden layer.
      logit = W2' * h_output + repmat(b2, 1, num_valid_cases);  % Input to output layer.
      prediction = 1 ./ (1 + exp(-logit));  % Output prediction.
      valid_CE = -mean(mean(sum(valid_labels .* log(prediction') + ...
          (1 - valid_labels) .* log(1 - prediction'), 2)));
      valid_classErr = 1 - sum(sum(valid_labels .* log_to_binary(prediction'), 2)) / double(num_valid_cases);
      
      %%%%%% Print out summary statistics at the end of the epoch %%%%%
      total_epochs = total_epochs + 1;
      train_CE_list(1, epoch) = train_CE;
      train_CE_list(2, epoch) = train_classErr;
      valid_CE_list(1, epoch) = valid_CE;
      valid_CE_list(2, epoch) = valid_classErr;
      if (mod(total_epochs, 1) == 0 || total_epochs < 11)
        fprintf(1,'%d  Train CE=%f, err=%f Valid CE=%f, err=%f\n',...
                total_epochs, train_CE, train_classErr, valid_CE, valid_classErr);
      end
    end

   
