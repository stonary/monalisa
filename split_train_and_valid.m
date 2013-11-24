function [train_inputs, train_targets, valid_inputs, valid_targets] = ...
    split_train_and_valid(inputs, targets, ratio)

    total_size = size(inputs, 2);
    train_size = floor(total_size * ratio);
    
    train_inputs = inputs(:, 1:train_size);
    valid_inputs = inputs(:, (train_size+1):total_size);

    train_targets = targets(1:train_size);
    valid_targets = targets((train_size+1):total_size);