function[inputs, targets] = merge_with_otherdata(inputs, targets, ...
    inputs2, targets2, num)
    
    [inputs2 targets2] = shuffle_data(inputs2, targets2);
    inputs = [inputs inputs2(:, 1:num)];
    targets = [targets; targets2(1:num, :)];
    