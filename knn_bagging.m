function prediction = knn_bagging(train, train_labels, data)

    addpath('./kNN_baseline');

    valid = double(data);
    if (ndims(valid) == 3)
        valid = double(reshape(valid,32*32,size(valid, 3)));
    end
    
    sets = 7; %[3, 5, 7, 10, 20, 30]
    num_per_set = 2000; %[1000, 1500, 2000, 2300]

    %prediction = zeros(size(test_images, 3), sets); %test set
    prediction = zeros(size(valid, 2), sets); %val set

    index = zeros(size(train, 2), sets);
    for i = 1:sets

        %index = randperm(size(tr_images, 3)); % test
         index(:, i) = randperm(size(train, 2)); % valid

%        sample = tr_images(:, :, index(1:num_per_set));
%        prediction(:, i) = knn_classifier(bestK, tr_images(:, :, index(1:num_per_set)), tr_labels(index(1:num_per_set)), test_images); %test
        prediction(:, i) = knn_classifier(5, train(:, index(1:num_per_set, i)), train_labels(index(1:num_per_set, i)), valid); %valid
    end

    prediction = transpose(mode(prediction, 2));
    
    save('test.mat', 'prediction');
end

