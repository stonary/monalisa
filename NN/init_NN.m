%% get the digit data
clear all
addpath('./..');
load labeled_data.mat;
load val_images;
% load test_images;

% Resize 32x32x2925 matrix to 1024x2925 matrix
resized = reshape(tr_images, 32*32, 2925);

%% sort data by expression
exps = repmat(struct('inputs', 1, 'labels', 1, 'other_inputs', 1, ...
    'other_labels', 1), 7, 1);
start_indices = zeros(8, 1);

[sorted_train_labels, order] = sort(tr_labels);
sorted_train = resized(:, order);
num_of_cases = size(sorted_train, 2);

for i = 1:7
    start_indices(i) = find(sorted_train_labels == i, 1);
end
start_indices(8) = size(sorted_train_labels, 1) + 1;

for i = 1:7
    exps(i).inputs = sorted_train(:, start_indices(i):(start_indices(i + 1) - 1));
    exps(i).labels = sorted_train_labels( start_indices(i):(start_indices(i + 1) - 1));
    exps(i).other_inputs = [sorted_train(:, 1: start_indices(i) - 1) ...
        sorted_train(:, start_indices(i+1): num_of_cases)];
    exps(i).other_labels = [sorted_train_labels(1: start_indices(i) - 1, :); ...
        sorted_train_labels(start_indices(i+1): num_of_cases, :)];
end

%% Train for each expression
% an array of struct
nns = repmat(struct('face', 0, 'ratio', 1, 'hidden', 1, 'eps', 1, 'mom', 1, 'epochs', 1, ...
    'w1', 1, 'b1', 1, 'w2', 1, 'b2', 1, 'CE', 1, 'ER', 1), 7, 1);
scores = repmat(struct('scores', 1), 7, 1);
for i = 1:7
    nn = nns(i);
    nn.face = i;
    % make sure the positive case : neg case ~= 1:1
    num_neg = size(exps(i).inputs, 2);
    if (size(exps(i).inputs, 2) > size(exps(i).other_inputs, 2))
        num_neg = size(exps(i).other_inputs, 2);
    end
    [train, labels] = merge_with_otherdata(exps(i).inputs, exps(i).labels, ...
        exps(i).other_inputs, exps(i).other_labels, num_neg);
    % change the label to be 0-1 for give expression
    labels = (labels == i);
    % shuffle
    [train, labels] = shuffle_data(train, labels);
    
    ratio = 0.5;
    % split into training set and validation set
    [train_inputs, train_targets, valid_inputs, valid_targets] = ...
        split_train_and_valid(train, labels, ratio);
    
    %% train for particular expression
    num_epochs = 8000;
    min_classErr = 1;   maxepoch = 0;    maxeps = 0;
    max_hidden = 0; max_mom = 0;
    %hiddens, eps, momentum, err/epoch
    scoring_materix = zeros(12 * 21 * 6, 5);
    k = 1;
%     num_hiddens = 50;     eps = 0.001;     momentum = 0;
    for num_hiddens = 100 %[10 50 100 200]%[10 20:20:200 500]
        for eps = 0.0005 %[0.001 0.005 0.01 0.05 0.1]%[0.001 0.005:0.005:0.1]
            for momentum = 0 %[0 0.001 0.01 0.1]%[0 0.01 0.05:0.05:0.2]
                fprintf('====== training: hiddens %d, eps %f, mom %f\n', num_hiddens, eps, momentum);
                
                [train_CE_list, valid_CE_list, W1, b1, W2, b2] = ...
                    train_NN(train_inputs, valid_inputs, train_targets, valid_targets, ...
                    num_hiddens, num_epochs, momentum, eps);
                % find the epoch with the minimal err
                [classErr, epoch] = min(valid_CE_list(2, :));
               
                % now do it again with the minimal err epoch
                fprintf('--------- retraining for epochs: %d ----------\n', epoch);
                [train_CE_list, valid_CE_list, W1, b1, W2, b2] = ...
                    train_NN(train_inputs, valid_inputs, train_targets, valid_targets, ...
                    num_hiddens, epoch, momentum, eps);
                final_err = valid_CE_list(2, epoch);
                
                if (final_err < min_classErr)
                    min_classErr = final_err;
                    maxepoch = epoch;  maxeps = eps; max_hidden = num_hiddens;
                    max_mom = momentum;
                    bestw1 = W1; bestb1 = b1; bestw2 = W2; bestb2 = b2;
                end
                scoring_materix(k, :) = [num_hiddens, eps, momentum, epoch, classErr];
                k = k + 1;
            end
        end
    end
    scores(i).scores = scoring_materix;
    % load the best into nns
    nn.ratio = ratio;
    nn.hidden = max_hidden; nn.eps = maxeps; nn.mom = max_mom;
    nn.epochs = maxepoch; nn.w1 = bestw1; nn.b1= bestb1; nn.w2 = bestw2; nn.b2 = bestb2;
    nn.ER = min_classErr;
    nns(i) = nn;
end

%% now train for the whole set
[train, labels] = shuffle_data(resized, tr_labels);
prediction = zeros(7, size(train, 2));
for i = 1:7
    nn = nns(i);
    prediction(i, :) = test_NN(train, nn.w1, nn.b1, nn.w2, nn.b2);
end

preds = zeros(size(train, 2), 1);
for i = 1:size(train, 2)
    [maxprob, k] = max(prediction(:, i));
    preds(i) = k;
end

err_rate = 1 - sum(labels == preds)/size(preds, 1);
fprintf('err_rate = %f', err_rate);

%% test
test_inputs = reshape(val_images, 32*32, 418);;
test_prediction = zeros(7, size(test_inputs, 2));
for i = 1:7
    nn = nns(i);
    test_prediction(i, :) = test_NN(test_inputs, nn.w1, nn.b1, nn.w2, nn.b2);
end

test_preds = zeros(size(test_inputs, 2), 1);
for i = 1:size(test_inputs, 2)
    [maxprob, k] = max(test_prediction(:, i));
    test_preds(i) = k;
end

% Fill in the test labels with 0 if necessary
if (length(test_preds) < 1253)
  test_preds = [test_preds; zeros(1253-length(test_preds), 1)];
end

% Print the predictions to file
fprintf('writing the output to prediction.csv\n');
fid = fopen('prediction.csv', 'w');
fprintf(fid,'%s,%s\n', 'Id','Prediction');
for i=1:length(test_preds)
  fprintf(fid,'%d,%d\n', i,test_preds(i));
end
fclose(fid);

% %% initialize the net parameters.
% num_hiddens = 10;
% eps = 0.01;       % the learning rate
% momentum = 0.0;   % the momentum coefficient
% num_epochs = 100; % number of learning epochs (number of passes through
%                   % the training set) each time runbp is called.
% %% training
% [train_CE_list, valid_CE_list, W1, b1, W2, b2] = ...
%     train_NN(train_inputs, valid_inputs, train_labels, valid_labels, ...
%     num_hiddens, num_epochs, momentum, eps);
%
% %% plotting
% plot_CE(train_CE_list(1, :), valid_CE_list(1, :), ...
%         'Epoch', 'Cross Entropy', 'Cross Entropy vs Epoch');
% plot_CE(train_CE_list(2, :), valid_CE_list(2, :), ...
%         'Epoch', 'Classification Error', 'Classification Error vs Epoch');

