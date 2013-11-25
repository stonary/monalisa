function[new_labels] = log_to_binary(labels)
% Given a label with continous value, make it 1-0 based on the highest 
% values

new_labels = zeros(size(labels));
for i = 1:size(labels, 1)
    new_labels(i, :) = (labels(i, :) == max(labels(i, :)));
end