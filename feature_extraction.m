clear,clc, close all
datapath='D:\Irfan_Haider\new_paper\Datasets\split - Copy\Test\Wheat';
imds=imageDatastore(datapath,  'IncludeSubfolders',true, 'LabelSource','foldernames');
total_split=countEachLabel(imds)
[imdsTrain,imdsTest] = splitEachLabel(imds,.5,'randomized');
numClasses = numel(categories(imdsTrain.Labels))


net=load('newNF.mat');
net=net.newNF;

net.Layers(1)
net.Layers(end)
% Number of class names for ImageNet classification task
imageSize = net.Layers(1).InputSize;
augmentedTrainingSet = augmentedImageDatastore(imageSize, imdsTrain, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, imdsTest, 'ColorPreprocessing', 'gray2rgb');

featureLayer = 'NewFc';

% trainFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
%     'MiniBatchSize', 32, 'OutputAs', 'columns');
% trainLabels = imdsTrain.Labels;

testFeaturesself = activations(net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
testLabelss = imdsTest.Labels;



 save('testFeaturesself','testFeaturesself');
save('testLabelss','testLabelss');


feat=im2double(testFeaturesself);
 feat=feat';
feat=array2table(feat);
% 
feat.type=testLabelss;
classificationLearner