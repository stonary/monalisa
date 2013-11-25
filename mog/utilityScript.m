load labeled_data.mat

% Resize 32x32x2925 matrix to 1024x2925 matrix
resized = reshape(tr_images,32*32,2925);

% Shuffle the data and labels
permutation = randperm(size(resized, 2));
resized_shuffle = resized(:, permutation);
labels_shuffle = tr_labels(permutation);

% Split into train and valid
train = resized_shuffle(:, 1:2500);
valid = resized_shuffle(:, 2501:2925);

train_labels = labels_shuffle(1:2500);
valid_labels = labels_shuffle(2501:2925);

% ------------------------------------------

start_indices = zeros(8, 1);

% sort data by expression
[sorted_train_labels, order] = sort(train_labels);
train_sorted = train(:, order);

for i = 1:7
    start_indices(i) = find(sorted_train_labels == i, 1);
end
start_indices(8) = size(sorted_train_labels, 1) + 1;

% Validation set
% ------------------------------------------

% sort valid data by expression
start_indices_valid = zeros(8, 1);

% sort data by expression
[sorted_valid_labels, order] = sort(valid_labels);
valid_sorted = valid(:, order);

for i = 1:7
    start_indices_valid(i) = find(sorted_valid_labels == i, 1);
end
start_indices_valid(8) = size(sorted_valid_labels, 1) + 1;


% ------------------------------------------

num_clusters = 50; % try 1, 10, 30, 40, 60, 80, 100, 200
% 80 -> 1142
% 60 -> 996
% 30 -> 782
% 40 -> 832
% 1 -> 280

%new
%150->72/425


num_iterations = 6;

pD = zeros(num_clusters, 7); % 7 because 7 expressions
muD = zeros(1024, num_clusters, 7);
varyD = zeros(1024, num_clusters, 7);
logProbXd = zeros(num_iterations, 7);

for i = 1:7
    [pD(:, i), muD(:, :, i), varyD(:, :, i), logProbXd(:, i)] = ...
        mogEM_kmeans( ...
            double(train_sorted(:, (start_indices(i)):(start_indices(i+1) - 1))), ...
            num_clusters, ...
            num_iterations, ...
            0.01); % 0.01 is min variance
end

% Classify the training set
%
n = start_indices(2:8) - start_indices(1:7); % number of example i in the training set

n = ones(1, 7);

num_correct = 0;
for i = 1:7
    x = double(train_sorted(:, (start_indices(i)):(start_indices(i+1) - 1)));
    [pi, class] = classifyMog(pD,muD,varyD,n,x);
    
    num_correct = num_correct + sum(class == i);
end

fprintf('num_correct: %d\n', num_correct);
%

% Classify the validation set
%{
n = start_indices_valid(2:8) - start_indices_valid(1:7); % number of example i in the training set

% TODO: Check whether can just use this instead of having different priors
n = ones(1, 7);

num_correct_valid = 0;
for i = 1:7
    x = double(valid_sorted(:, (start_indices_valid(i)):(start_indices_valid(i+1) - 1)));
    [pi, class] = classifyMog(pD,muD,varyD,n,x);
    
    num_correct_valid = num_correct_valid + sum(class == i);
end

fprintf('num_correct_valid: %d\n', num_correct_valid);
%}