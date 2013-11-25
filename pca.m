load digits;

train = [train2, train3]; % 256 x 600 data matrix
validSet = [valid2, valid3];

% generate matrix containing correct outputs
validLabels = [zeros(100, 1); ones(100, 1)];

% values of m to try
mVec = [2, 5, 10, 20];

% counter for the for loop
count = 1;

% vector to store the error rate for each value of m
fracError = zeros(4, 1);

for m = mVec

    C = zeros(256, 256);
    N = size(train, 2);
    
    % find the mean vector
    theMean = mean([train2, train3], 2);

    % find the matrix C that we will find eigenvectors from
    for i = 1:N
       sub = train(:, i) - theMean;
       C = C + ((1/N) * sub) * transpose(sub);
    end

    % find eigenvectors and eigenvalues
    [eigVec, eigVal] = eig(C);

    % store indices of sorted eigenvalues
    [a, indices] = sort(diag(eigVal), 'descend');

    % sort eigenvectors according to indices
    eigVec = eigVec(:, indices);

    % 256 x m matrix
    U = eigVec(:, 1:m);

    % find projected data which is m x 600
    projTrain = transpose(U) * train;

    % find projected validation set
    projValid = transpose(U) * validSet;

    % Prepare to feed to run_knn function from A1 which runs k-NN
    knnPredict = run_knn(1, transpose(projTrain), [zeros(300, 1); ones(300, 1)], transpose(projValid));

    % find how many correct we have
    numCorrect = sum(knnPredict == validLabels);

    % find the error rate
    fracError(count) = (200 - numCorrect) / 200;
    
    count = count + 1;
end

clf
  hold on, ...
  plot(mVec, fracError, 'r'),...
  %legend('Train', 'Valid', 'Test'),...
  title('Classification Error vs. Number of Principal Components'), ...
  xlabel('Number of Principal Components/Eigenvectors'), ...
  ylabel('Classification Error');