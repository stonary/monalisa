function[new_labels] = dimup(labels, dim)

% n x 1 labels with dim number of values -> n x dim labels with 0 and 1's

new_labels = zeros(size(labels, 1), dim);
for i = 1:size(labels, 1)
    new_labels(i, labels(i)) = 1;
end
