function class_label = train_centroidSubClassifier(cluster_vectors, data, labels)
    
    comp_centroids = train_centroidClassifier(data, labels);
    dis = 0; %Distance
    mDis = 0; %Minimum distance
    sizeOfK = size(cluster_vectors);
    sizeOflabels = size(unique(labels));
    class_label = zeros(1,sizeOfK(2));
    
    for k = 1:sizeOfK(2) %Label the classes from k-means
        for km = 1:sizeOflabels(1)
            dis = norm(cluster_vectors(:,k) - comp_centroids(:,km)); %Find label for kmeans class
            if  dis < mDis || km == 1
                mDis = dis;
                class_label(:,k) = km-1;
            end
        end
    end
end

