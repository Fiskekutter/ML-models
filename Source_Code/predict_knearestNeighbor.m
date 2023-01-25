function [KNNPrediction, KNNacc] = predict_knearestNeighbor(neighbors,train_data, test_data, test_labels, train_labels)
KNum = neighbors; %Number of neighbors
[KNNpredict,~] = knnsearch(train_data', test_data',"K", KNum); %Matlab KNN search
KNNPrediction = zeros(size(test_labels,1),1);

for l = 1:size(KNNpredict,1)
    for i = 1:size(KNNpredict,2)
        KNNpredict(l,i) = train_labels(KNNpredict(l,i),1); %Replace every index with the prediction label
    end
    [GC, GR] = groupcounts(KNNpredict(l,:)'); %Gets the number of labels for neighbors
    KNNPrediction(l,1) = GR(1,1); %Sets the prediction to the highest number of labels for neighbors
end

KNNacc = 0; %Calculate accuracy/score
for i = 1:size(KNNPrediction,1)
    if KNNPrediction(i) == test_labels(i)
        KNNacc = KNNacc + 1;
    end
end
KNNacc = (KNNacc/size(KNNPrediction,1))*100; %Accuracy
end

