% load digit dataset
digitDatasetPath = fullfile('D:\Irfan_Haider\new_paper\Datasets\split\Train\Wheat');
 imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
 [imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');
%% design CNN
 net=load('updatedvit.mat'); 
net=net.updatedvit;


% net=vgg19;
% net=net.vgg19;
 lgraph=layerGraph(net);



numClasses = numel(categories(imdsTrain.Labels));    
newFCLayer = fullyConnectedLayer(numClasses,'Name','NewFc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,'head',newFCLayer);
newClassLayer = softmaxLayer('Name','NewSoftmax');
lgraph = replaceLayer(lgraph,'softmax',newClassLayer);

newClassLayer1 = classificationLayer('Name','classification');
lgraph = addLayers(lgraph,newClassLayer1);
%lgraph = replaceLayer(lgraph,'output',newClassLayer1);
lgraph = connectLayers(lgraph,'NewSoftmax','classification');

  %% Augmenter
    augmenter = imageDataAugmenter( ...
        'RandRotation',[-5 5],'RandXReflection',1,...
        'RandYReflection',1,'RandXShear',[-0.05 0.05],'RandYShear',[-0.05 0.05]);
    %% Resize training and testing data according to network
    auimds = augmentedImageDatastore([384 384 3],imdsTrain,'ColorPreprocessing','gray2rgb','DataAugmentation',augmenter);
    auimdsVali = augmentedImageDatastore([384 384 3],imdsValidation,'ColorPreprocessing','gray2rgb','DataAugmentation',augmenter);

 options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MaxEpochs',10,'MiniBatchSize',16,...
        'Shuffle','every-epoch', ...
        'ValidationData',auimdsVali,...
        'InitialLearnRate',0.000265, ...
        'ValidationFrequency', 5, ...
        'Verbose',false, ...
        'Plots','training-progress');
% set training options


% training the network
% TrainedModifiedNet16 = trainNetwork(auimds, lgraph, options);
newNF = trainNetwork(auimds, lgraph, options);

% save('TrainedModifiedNet16','TrainedModifiedNet16');
save('newNF','newNF');