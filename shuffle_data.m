function [inputs_shuffled, targets_shuffled] = shuffle_data(inputs, targets)

    % Shuffle the data and labels
    permutation = randperm(size(inputs, 2));
    inputs_shuffled = inputs(:, permutation);
    targets_shuffled = targets(permutation, :);