function [NSC_Predictions, Accuracy] = predict_centroidSubClassifier(class_label, cluster_vectors, data, labels)
        
    %Use NCC with sub classes    
    %[NSC_Predictions, nsc_acc] = predict_centroidClassifier(cluster_vectors, data, labels);
    %sizeOftestLabels = size(labels);
rightanswers = 0;
minDist = 0;
labelPredict = 0;

for k = 1:size(labels,1)
        for i = 1:size(cluster_vectors,2)
            distance = norm(cluster_vectors(:,i) - data(:,k)); %Euclidean Distance Between class centroids and data
            if distance < minDist || i == 1 %The less distance the more likely it is the right class
                minDist = distance;
                labelPredict = class_label(1,i);
            end
        end 
        if labelPredict == labels(k)
            rightanswers = rightanswers + 1;
        end
end
    Accuracy = (rightanswers/size(labels,1))*100;
    NSC_Predictions = rightanswers;





    
%     right_answers = 0;
%     for per = 1:sizeOftestLabels(1)
%         num = NSC_Predictions(:,per)+1;
%         NSC_Predictions(:,per) = class_label(:,num);
%         if NSC_Predictions(:,per) == labels(per,:)
%             right_answers = right_answers+1;
%         end
%     end
%     nsc_acc = (right_answers/sizeOftestLabels(1))*100;
%     Accuracy = nsc_acc;
%     unique(NSC_Predictions);
%     unique(class_label);
%     NSC_Predictions;
end