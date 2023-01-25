%% Train NCC
function centroids = train_centroidClassifier(data, label)    
    sizeOfData = size(data);
    sizeOfLabels = size(label);
    labels = unique(label);
    uniqLables = size(labels);
    summations = zeros(sizeOfData(1), uniqLables(1));
    means = zeros(sizeOfData(1), uniqLables(1));
    sizeOfsum = 0;

    for i = 1:size(labels)
        for j = 1:sizeOfLabels(1)
            if label(j) == labels(i)
                summations(:,i) = summations(:,i) + data(:,j);
                sizeOfsum = sizeOfsum + 1;
            end  
        end
    means(:,i) = summations(:,i)/sizeOfsum;
    sizeOfsum = 0;
    end
    centroids = means;
end