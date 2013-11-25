%% get the digit data
clear all
addpath('./..');
load labeled_data.mat;
load val_images;
% load test_images;

% Resize 32x32x2925 matrix to 1024x2925 matrix
resized = reshape(tr_images, 32*32, 2925);
train = resized;
labels = dimup(tr_labels, 7);
% scores = repmat(struct('scores', 1), 7, 1);
%% Train data

[train, labels] = shuffle_data(train, labels);
ratio = 0.7;
% split into training set and validation set
[train_inputs, train_targets, valid_inputs, valid_targets] = ...
    split_train_and_valid(train, labels, ratio);
    
%% train
num_epochs = 4000;

% initialize
min_classErr = 1;   maxepoch = 0;    maxeps = 0;
max_hidden = 0; max_mom = 0;
k = 1;
scores = repmat(struct('scores', 1), 7, 1);
%     num_hiddens = 50;     eps = 0.001;     momentum = 0;
for num_hiddens = 7000%[50:10:100]%[10 20:20:200 500]
    for eps = 0.0001%[0.001 0.005]% 0.01 0.05 0.1]%[0.001 0.005:0.005:0.1]
        for momentum = 0.1 %[0 0.001 0.01 0.1]%[0 0.01 0.05:0.05:0.2]
            %hiddens, eps, momentum, err/epoch
            scoring_materix = zeros(12 * 21 * 6, 5);
            fprintf('====== training: hiddens %d, eps %f, mom %f\n', num_hiddens, eps, momentum);
            
            [train_CE_list, valid_CE_list, W1, b1, W2, b2] = ...
                train_NN_sevenOutputs(train_inputs, valid_inputs, train_targets, valid_targets, ...
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

%% now train for the whole set
[train, labels] = shuffle_data(resized, tr_labels);
labels = dimup(labels, 7);

prediction = test_NN(train, bestw1, bestb1, bestw2, bestb2);

err_rate = 1 - sum(labels .* prediction)/size(prediction, 1);
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

