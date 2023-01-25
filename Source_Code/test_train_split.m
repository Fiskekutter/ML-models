function [test_data,test_label,train_data,train_label] = test_train_split(split, data, label)
q = [label'; data]; %So the random divide happens on data and labels

[train,~,test] = dividerand(q, split, 0, 1-split); %Randomly split label and data

train_label = train(1,:)'; %Collect train labels
test_label = test(1,:)'; %Collect test labels
train_data = train(2:size(train,1),:); %Collect train data
test_data = test(2:size(test,1),:); %Collect test data

end