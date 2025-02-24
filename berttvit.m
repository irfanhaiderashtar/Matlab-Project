%% DatasetLoad

%% import Dataset
digitDatasetPath = fullfile('D:\Irfan_Haider\new_paper\Datasets\split\Train\Wheat');
 imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
 %% split data into training and validation
% numTrainFiles = 0.8;
% [imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,"randomize");
[imdsTrain,imdsValidation,imdsTest] = splitEachLabel(imds,0.7,0.15);
%% Visualization of Dataset Random 6 samples
% num_images=length(imds.Labels);
% perm=randperm(num_images,6);
% for idx=1:length(perm)
% 
%     subplot(2,3,idx);
%     imshow(imread(imds.Files{perm(idx)}));
%     title(sprintf('%s',imds.Labels(perm(idx))))
% 
% end
%% View the ClassesName
classNames = categories(imds.Labels);
numClasses = numel(categories(imds.Labels))
%% Vision Transformer
% net = visionTransformer("base-16-imagenet-384");
net=updatedvit;
lgraph = layerGraph(net); % Extract the layer graph from the trained network and plot the layer graph.
 net.Layers(1); % Getting 1st layer information
inputSize = net.Layers(1).InputSize; % getting image size info
lgraph = removeLayers(lgraph, {'cls_index','head','softmax'});

newLayers = [  
    globalAveragePooling1dLayer('Name','gap1')
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];

lgraph = addLayers(lgraph,newLayers);
 lgraph = connectLayers(lgraph,'encoder_norm' ,'gap1');
%% Augmentor

augmenter = imageDataAugmenter( ...
    RandXReflection=true, ...
    RandRotation=[-90 90], ...
    RandScale=[1 2]);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,DataAugmentation=augmenter);

%% %% apply augmentator
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);
%% training Options 
miniBatchSize = 6;

numObservationsTrain = numel(augimdsTrain.Files);
 numIterationsPerEpoch = floor(numObservationsTrain/miniBatchSize);

options = trainingOptions("adam", ...
    'ExecutionEnvironment','gpu',...
    MaxEpochs=10, ...
    InitialLearnRate=0.0001, ...
    MiniBatchSize=miniBatchSize, ...
    ValidationData=augimdsValidation, ...
    ValidationFrequency=numIterationsPerEpoch, ...
    OutputNetwork="best-validation-loss", ...
    Plots="training-progress", ...
    Verbose=false);

%% Train network

  SYS_VIT=trainNetwork(augimdsTrain,lgraph,options);
  % netTransformerbert = trainnet(augimdsTrain,lgraph,"crossentropy",options);
save('SYS_VIT','SYS_VIT');
