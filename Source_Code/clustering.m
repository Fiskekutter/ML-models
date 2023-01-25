function [classes_ordered,classes_ordered_label] = clustering(train_data, train_label, clusters)
%CLUSTERING Summary of this function goes here
%   Detailed explanation goes here
data_order = [];
label_order = [];
classes_ordered = [];
classes_ordered_label = [];
unique_data = unique(train_label); %Test labels since it's smaller
k_order = clusters; %Kmeans classes

%Split data into its classes
for uniq = 1:size(unique_data,1)
    for data = 1:size(train_label,1)
        if train_label(data,1) == unique_data(uniq,1)
            data_order(:,data) = train_data(:,data);
            label_order(data,:) = train_label(data,:);
        end
    end
    %Kmeans on each class
    [~, k_n] = kmeans(data_order', k_order);
    k_n = transpose(k_n);
    for klm = 1:k_order
        classes_ordered = [classes_ordered k_n(:, klm)];
        classes_ordered_label = [classes_ordered_label unique_data(uniq,1)]; 
    end
    data_order = []; %reset because the data could have remains of the previous
    label_order = []; %reset because the labels could have remains of the previous
end
end

