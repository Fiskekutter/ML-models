function predict_labels = test_nearestneighbor(train_data, test_data, train_label)
%TEST_NEARESTNEIGHBOR Summary of this function goes here
%   Detailed explanation goes here
    sizeOftest = size(test_data);
    sizeOftrain = size(train_data);
    sizeOflabel = size(train_label);
    distance = 0;
    minDistance = 0;
    predict_labels = zeros(sizeOftest(2),1);

    for i = 1:sizeOftest(2)
        for km = 1:sizeOftrain(2)
            distance = norm(test_data(:,i) - train_data(:,km));
            if  distance < minDistance || km == 1
                minDistance = distance;
                predict_labels(i,1) = train_label(km,1);
            end
        end
    end    
end

