clc;
clear; 
close all;

%% Parameters
inputSize = 64; 
batchSize = 16;
numEpochs = 10;

%% Load and Prepare Dataset
datasetDir = 'preprocessed_dataset/';
noTumorDir = fullfile(datasetDir, 'no');
yesTumorDir = fullfile(datasetDir, 'yes');

% Get image files
noTumorImages = dir(fullfile(noTumorDir, '*.jpg'));
yesTumorImages = dir(fullfile(yesTumorDir, '*.jpg'));

% Initialize arrays
dataset = [];
labels = [];

% Load No Tumor images
for i = 1:length(noTumorImages)
    imgPath = fullfile(noTumorDir, noTumorImages(i).name);
    img = imread(imgPath);
    
    % Convert to RGB if needed
    if size(img, 3) == 1
        img = cat(3, img, img, img); 
    elseif size(img, 3) > 3
        img = img(:,:,1:3); 
    end
    
    img = imresize(img, [inputSize, inputSize]);
    dataset = cat(4, dataset, img); 
    labels = [labels; 0]; 
end

% Load Yes Tumor images
for i = 1:length(yesTumorImages)
    imgPath = fullfile(yesTumorDir, yesTumorImages(i).name);
    img = imread(imgPath);
    
    % Convert to RGB if needed
    if size(img, 3) == 1
        img = cat(3, img, img, img);
    elseif size(img, 3) > 3
        img = img(:,:,1:3);
    end
    
    img = imresize(img, [inputSize, inputSize]);
    dataset = cat(4, dataset, img);
    labels = [labels; 1]; 
end

% Convert to appropriate data types
dataset = single(dataset); 
labels = categorical(labels, [0, 1], {'NoTumor', 'Tumor'});

% Normalize pixel values to [0, 1]
dataset = dataset ./ 255;

%% Split into Training and Test Sets
rng(0); % For reproducibility
cv = cvpartition(numel(labels), 'HoldOut', 0.2);

trainIdx = training(cv);
testIdx = test(cv);      

xTrain = dataset(:,:,:,trainIdx);
yTrain = labels(trainIdx);

xTest = dataset(:,:,:,testIdx);
yTest = labels(testIdx);

%% Create CNN Model
layers = [
    imageInputLayer([inputSize inputSize 3])
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 64, 'Padding', 'same')
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    
    flattenLayer()
    fullyConnectedLayer(64)
    reluLayer()
    dropoutLayer(0.3)
    fullyConnectedLayer(2)
    softmaxLayer()
    classificationLayer()
];

%% Training Options
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', numEpochs, ...
    'MiniBatchSize', batchSize, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {xTest, yTest}, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%% Train the Model
[net, info] = trainNetwork(xTrain, yTrain, layers, options);

%% Evaluate the Model
% Training accuracy
trainPred = classify(net, xTrain);
trainAccuracy = mean(trainPred == yTrain);
fprintf('Training Accuracy: %.2f%%\n', trainAccuracy*100);

% Test accuracy
testPred = classify(net, xTest);
testAccuracy = mean(testPred == yTest);
fprintf('Test Accuracy: %.2f%%\n', testAccuracy*100);

%% Confusion Matrix
figure;
confusionchart(yTest, testPred);
title('Confusion Matrix');

%% Classification Report
cm = confusionmat(yTest, testPred);

precision = cm(2,2) / (cm(2,2) + cm(1,2));
recall = cm(2,2) / (cm(2,2) + cm(2,1));
f1Score = 2 * (precision * recall) / (precision + recall);

fprintf('\nClassification Metrics:\n');
fprintf('Precision: %.4f\n', precision);
fprintf('Recall: %.4f\n', recall);
fprintf('F1-Score: %.4f\n', f1Score);



%% Save the Model
save('BrainTumor10EpochsCategorical.mat', 'net');