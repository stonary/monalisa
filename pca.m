
%Input:
%   data - a mxn matrix, where n is the number of cases,
%            m is the dimension of each case
%   new_dimensions - the number of dimensions of the new data.
%                        must be <= m
%
%Output:
%   projected_data - a cxn matrix, where c = new_dimensions
%
function [projected_data] = pca(data, new_dimensions)

    C = zeros(size(data, 1), size(data, 1));
    N = size(data, 2);

    double_data = double(data);
    
    % find the mean vector
    theMean = mean(data, 2);

    % find the matrix C that we will find eigenvectors from
    for i = 1:N
       sub = double_data(:, i) - theMean;
       C = C + ((1/N) * sub) * transpose(sub);
    end

    % find eigenvectors and eigenvalues
    [eigVec, eigVal] = eig(C);

    % store indices of sorted eigenvalues
    [a, indices] = sort(diag(eigVal), 'descend');

    % sort eigenvectors according to indices
    eigVec = eigVec(:, indices);

    % 256 x m matrix
    U = eigVec(:, 1:new_dimensions);

    % find projected data
    projected_data = transpose(U) * double_data;

end

