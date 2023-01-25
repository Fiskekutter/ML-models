%% Predict NCC
function [guesses, acurracy] = predict_centroidClassifier(centroids, predict_data, predict_lables)
    lable_size = size(predict_lables);
    predictions = zeros(1, lable_size(1));
    numberOfClasses = size(centroids);
    uniq = unique(predict_lables)
    labelPredict = 0;
    minDist = 0;
    distance = 0;
    rightanswers = 0;

    for k = 1:lable_size(1)
        for i = 1:numberOfClasses(2)
            distance = norm(centroids(:,i) - predict_data(:,k)); %Euclidean Distance Between class centroids and data
            if distance < minDist || i == 1 %The less distance the more likely it is the right class
                minDist = distance;
                labelPredict = uniq(i,1);
            end
        end
        predictions(k) = labelPredict; 
        if predictions(k) == predict_lables(k)
            rightanswers = rightanswers + 1;
        end
    end
    guesses = predictions;
    acurracy = (rightanswers/size(predict_lables,1))*100;
end

