%% Import MNIST dataset
clear all;

train_images = loadMNISTImages('train-images.idx3-ubyte');
test_images = loadMNISTImages('t10k-images.idx3-ubyte');
train_labels = loadMNISTLabels('train-labels.idx1-ubyte');
test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');

%% PCA Mnist
[pca_train_mnist, ~] = pca(train_images, 'NumComponents', 2);

[pca_test_mnist, ~] = pca(test_images, 'NumComponents', 2);

pca_train_mnist = transpose(pca_train_mnist);
pca_test_mnist = transpose(pca_test_mnist);

%% Import ORL dataset
data = load("orl_data.mat");
lbls = load("orl_lbls.mat");

orl_data = data.data; %Collect the data from nested structure
orl_lbls = lbls.lbls; %Collect the labels from nested structure

%% TestTrain Split ORL
clear section;
dot = [orl_lbls'; orl_data]; %Construct new set from orl data and label for splitting
check = 0;


while check == 0 %In case all classes are not represented in the split
    [orl_train,~,orl_test] = dividerand(dot, 0.7, 0, 0.3); %Use diverand for splitting dataset
    orl_train_lbls = orl_train(1,:)'; %Collect train labels from split dataset
    orl_test_lbls = orl_test(1,:)'; %Collect test labels from split dataset
    orl_train_images = orl_train(2:size(orl_train,1),:); %Collect train data from split dataset
    orl_test_images = orl_test(2:size(orl_test,1),:); %collect test data from split dataset
    if size(unique(orl_train_lbls),1) == 40 && size(unique(orl_test_lbls),1) == 40
        check = 1;
    end
end

%% PCA ORL
clear section;
[pca_train_orl, ~] = pca(orl_train_images, 'NumComponents', 2);
[pca_test_orl, ~] = pca(orl_test_images, 'NumComponents', 2);

pca_train_orl = transpose(pca_train_orl); %PCA data is transposed
pca_test_orl = transpose(pca_test_orl); %PCA data is transposed
%% PCA Scatter
%Plotting pca components
scatter(pca_coeff_mnist(:,1),pca_coeff_mnist(:,2));

%% TSNE
train_images_trans = transpose(orl_data);

y = tsne(train_images_trans,"Algorithm","exact","Distance","euclidean");

gscatter(y(:,1),y(:,2));

%% Using NCC on MNIST
classes = train_centroidClassifier(train_images, train_labels); %Calculate mean vectors/Centroids
[~, mnist_acc] = predict_centroidClassifier(classes, test_images, test_labels); %Predict
%score/accuracy
mnist_acc

%% NCC On MNIST PCA
classes = train_centroidClassifier(pca_train_mnist, train_labels); %Calculate mean vectors/Centroids
[~, pca_mnist_acc] = predict_centroidClassifier(classes, pca_test_mnist, test_labels); %Predict
%score/accuracy
pca_mnist_acc

%% Using NCC on ORL
classes = train_centroidClassifier(orl_train_images, orl_train_lbls);
[~, NCC_Orl_acc] = predict_centroidClassifier(classes, orl_test_images, orl_test_lbls); %Predict
NCC_Orl_acc %Accuracy changes because of random train test split

%% Using NCC on ORL pca
classes = train_centroidClassifier(pca_train_orl, orl_train_lbls); %Train
[predicts, NCC_Orl_pca_acc] = predict_centroidClassifier(classes, pca_test_orl, orl_test_lbls); %Predict
NCC_Orl_pca_acc %Changes based on train test split

%% NSC MNIST
clear section;
[mnist_cluster_classes, mnist_cluster_labels] = clustering(train_images, train_labels, 5);

%NSC prediction
[~, nsc_mnist_acc] = predict_centroidSubClassifier(mnist_cluster_labels, mnist_cluster_classes, test_images, test_labels);
nsc_mnist_acc

%% NSC ORL
clear section;
[orl_classes_ordered, orl_classes_ordered_label] = clustering(orl_train_images, orl_train_lbls, 2);

%NSC predict
[~, NSC_orl_accuracy] = predict_centroidSubClassifier(orl_classes_ordered_label, orl_classes_ordered, orl_test_images, orl_test_lbls);
NSC_orl_accuracy

%% NSC MNIST PCA
clear section;
[mnist_pca_cluster_classes, mnist_pca_cluster_labels] = clustering(pca_train_mnist, train_labels, 5);

%NSC prediction
[~, nsc_mnist_pca_acc] = predict_centroidSubClassifier(mnist_pca_cluster_labels, mnist_pca_cluster_classes, pca_test_mnist, test_labels);
nsc_mnist_pca_acc

%% NSC ORL PCA
clear section;
[orl_pca_cluster_classes, orl_pca_cluster_labels] = clustering(pca_train_orl, orl_train_lbls, 5);

%NSC prediction
[~, nsc_orl_pca_acc] = predict_centroidSubClassifier(orl_pca_cluster_labels, orl_pca_cluster_classes, pca_test_orl, orl_test_lbls);
nsc_orl_pca_acc

%% K-NN MNIST
[mnist_KNNPrediction, mnist_KNNacc] = predict_knearestNeighbor(15,train_images, test_images, test_labels, train_labels);
mnist_KNNacc

%% K-NN search ORL
[orl_KNNPrediction, orl_KNNacc] = predict_knearestNeighbor(15,orl_train_images, orl_test_images, orl_test_lbls, orl_train_lbls);
orl_KNNacc

%% K-NN orl PCA
[orl_pca_KNNPrediction, orl_pca_KNNacc] = predict_knearestNeighbor(15,pca_train_orl, pca_test_orl, orl_test_lbls, orl_train_lbls);
orl_pca_KNNacc

%% K-NN ORL MNIST
[mnist_pca_KNNPrediction, mnist_pca_KNNacc] = predict_knearestNeighbor(15,pca_train_mnist, pca_test_mnist, test_labels, train_labels);
mnist_pca_KNNacc

%% Show mnist and orl images
transpose_Mnist = train_images.'; %Transpose Mnist
transpose_ORL = orl_data.'; %Transpose ORL
img_MNIST = reshape(transpose_Mnist,[60000,28,28]); %Reshape to original structure MNIST
img_ORL = reshape(transpose_ORL,[400,40,30]); %Reshape to original structure ORL

figure(1);
for i = 1:12
    first_img = img_MNIST(i,:,:); %Picks the first image in MNIST
    first_img = reshape(first_img,[28,28]); %Reshape so the first image can be displayed
    subplot(3,4,i), imshow(first_img, 'InitialMagnification',500); %Shows the image
end
figure(2);
for i = 1:12
    scd_img = img_ORL(i*12,:,:); %Picks the first image in ORL
    scd_img = reshape(scd_img,[40,30]); %Rehsape so the first image can be displayed
    subplot(3,4,i), imshow(scd_img, 'InitialMagnification',500); %Shows the image
end

%% Built in toolbox function for plotting PCA data NOT USED
mapcaplot(train_images',train_labels)

%% 
mapcaplot(orl_data', orl_lbls)

%% SCATTER PLOT FOR PCA MNIST
clear section;
class_lb = [];
ord_lb = [];
un = unique(train_labels);

pca1_mn = pca_train_mnist(1,:);
pca2_mn = pca_train_mnist(2,:);

[ord_lb,isort] = sort(train_labels'); %Sort data to be in incrementing order
pca1_mn = pca1_mn(isort); %sort pca1
pca2_mn = pca2_mn(isort); %sort pca2

class_lb = [pca1_mn; pca2_mn]; %Concatenate arrays to be the rows of class_lb

%Colours for scatter
c = [1 0 0; 0 1 0; 0 0 1; 1 1 0; 0 1 1; 1 0 1; 0 0 0; 0.5 0.2 0; 0.8 0 0.6; 0 0.3 0.6];

cl = [];

b = []; %Numbers for legend
for t = 1:10
    b = [b t-1];
end
b = string(b);

[GR, GC] = groupcounts(ord_lb(1,:)');

figure(1)

for i = 1:size(un,1)
    for k = 1:size(train_labels,1)
        if ord_lb(k) == un(i,:)
            cl = [cl class_lb(:,k)];
        end
    end
    scatter(cl(1,:), cl(2,:), 10, c(i,:), 'filled');
    legend(b,'Location', 'Best');
    hold on
    cl = [];
end
xlabel("Principal component axis 1");
ylabel("Principal component axis 2");


%% Scatter plot for PCA ORL
clear section;
class_lb = [];
ord_lb = [];
un = unique(orl_train_lbls);

[pca_train_orl_scat, ~] = pca(orl_data, 'NumComponents', 2);

pca_train_orl_scat = transpose(pca_train_orl_scat); %PCA data is transposed

pca1_orl = pca_train_orl_scat(1,:);
pca2_orl = pca_train_orl_scat(2,:);

[ord_lb,isort] = sort(orl_train_lbls'); %Sort data to be in incrementing order
pca1_orl = pca1_orl(isort); %sort pca1
pca2_orl = pca2_orl(isort); %sort pca2

class_lb = [pca1_orl; pca2_orl]; %Concatenate arrays to be the rows of class_lb

%Colours for scatter
c = [1 0 0; 0 1 0; 0 0 1; 1 1 0; 0 1 1; 
    1 0 1; 0 0 0; 0.5 0.2 0; 0.8 0 0.6; 
    0 0.3 0.6; 0.2 0 0; 0.6 0.1 0; 0 0.5 0.2;
    0.7 0.7 0; 0.7 0.7 1; 0.2 0.2 1; 1 0.2 0.2;
    0.4 0.9 0; 0.4 0.9 1; 1 0.4 0.9; 1 1 0.2;
    0.2 0.2 0.2; 0 0.9 0.9; 0 1 0.2; 0 0.2 1;
    0.3 0.2 0.9; 0.4 0.6 0.1; 0.8 0.5 0.2; 0.1 0.8 0.8;
    0.8 0.8 0.1; 0.8 0.1 0.8; 0.3 0.8 0.3; 0.7 0.3 0.6;
    0 1 0.5; 1 1 0.5; 1 0 0.5; 0.5 1 1; 0.5 0 1; 0.3 0.1 0.4; 0 0 0];

cl = [];

b = []; %Numbers for legend
for t = 1:40
    b = [b t];
end
b = string(b);

[GR, GC] = groupcounts(ord_lb(1,:)');

figure(1)
xlabel("Principal component axis 1");
ylabel("Principal component axis 2");
for i = 1:size(un,1)
    for k = 1:size(orl_train_lbls,1)
        if ord_lb(k) == un(i,:)
            cl = [cl class_lb(:,k)];
        end
    end
    scatter(cl(1,:), cl(2,:), 15, c(i,:), 'filled');
    lgd = legend(b,'Location', 'Best');
    lgd.FontSize = 4;
    hold on
    cl = [];
end
