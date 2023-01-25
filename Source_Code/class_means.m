%% Kmeans
function classes = class_means(clusterNum, data, iterations)

    %K-means
    dataSize = size(data);
    clusters = clusterNum;
    %clusterCenters = kmeans(train_images,clusters*10);
    cluster_vectors = zeros(784,clusters);
    old_clusters = zeros(784,clusters);
    assigned_vectors_class = zeros(1,dataSize(2));
    clust_distance = 0;
    minDistance = 0;
    l_DP = 0;
    means_class = zeros(1,clusters);
    rng("shuffle","twister");
    
    r = randi([1 dataSize(2)],1,clusters); %Assign random numbers to cluster index
    r(randperm(length(r))); %Shuffle array
    uniqClusters = size(unique(r)); %Check if all numbers are unique
    
    while uniqClusters(2) ~= clusters %Check if any clusters are not the same
        r = randi([1 dataSize(2)],1,clusters); %Assign random numbers to cluster index
        r(randperm(length(r))); %Shuffle array
        uniqClusters = unique(r); %Check if all numbers are unique
    end
    
    for m = 1:clusters %Assign random cluster vectors
        cluster_vectors(:,m) = data(:,r(m));
    end
    
    new = 1;
    old = 0;
    max_it = iterations; %Max iterations of k-means
    it = 0;
    
    while new ~= old %Do while there is a change in class/cluster vectors
        
        it = it + 1;
        if it == max_it %Exit loop if max iterations have been reached
            old = 1;
        end
    
        if old_clusters == cluster_vectors %If there is no change in clusters/classes
            old = 1; %Exit loop
        end
    
        %Before evaluating new cluster vectors
        old_clusters = cluster_vectors; %Assign the old clusters to the newest found
    
        %Calculate new cluster vectors
        for i = 1:dataSize(2)
            for mn = 1:clusters
                clust_distance = norm(cluster_vectors(:,mn) - data(:,i)); %Euclidean Distance Between class centroids and data
                if  clust_distance < minDistance || mn == 1
                    minDistance = clust_distance;
                    assigned_vectors_class(:,i) = mn;
                    l_DP = mn;
                end
            end
            if minDistance ~= 0 % Case where the vector is the same as the class
                cluster_vectors(:,l_DP) = cluster_vectors(:,l_DP) + data(:,i); %Summa
                means_class(1,l_DP) = means_class(1,l_DP) + 1; %Used in calculating the mean vector
            end
        end
        
        for qr = 1:clusters %Calculate the mean class vectors for the clusters
            cluster_vectors(:,qr) = cluster_vectors(:,qr)/means_class(1,qr);
            means_class(1,qr) = 0;
        end
    end
    classes = cluster_vectors;
end