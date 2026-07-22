clc; clear; close all;


tumorDir = 'C:\Users\HP\OneDrive\Desktop\1911021_BME424_Project\Traditional\yes\';
nonTumorDir = 'C:\Users\HP\OneDrive\Desktop\1911021_BME424_Project\Traditional\no\';

%% Parameters
params.areaThreshold = 800;      
params.circularityThreshold = 0.7; 
params.solidityThreshold = 0.5;  

%% Process 10 Tumor (Yes) Images
tumorFiles = dir(fullfile(tumorDir, '*.jpg'));
fprintf('==== Processing Tumor (YES) Images ====\n');
for i = 1:min(10, length(tumorFiles))
    imgPath = fullfile(tumorDir, tumorFiles(i).name);
    processSingleImage(imgPath, params, i, 'Tumor');
end

%% Process 10 Non-Tumor (No) Images
nonTumorFiles = dir(fullfile(nonTumorDir, '*.jpg'));
fprintf('\n==== Processing Non-Tumor (NO) Images ====\n');
for i = 1:min(10, length(nonTumorFiles))
    imgPath = fullfile(nonTumorDir, nonTumorFiles(i).name);
    processSingleImage(imgPath, params, i, 'Non-Tumor');
end

%% ================ Core Processing Function ================
function processSingleImage(imgPath, params, imgNum, imgType)
    % Read and preprocess image
    inputImg = imread(imgPath);
    if size(inputImg, 3) == 3
        grayImg = rgb2gray(inputImg);
    else
        grayImg = inputImg;
    end
    
    % Enhanced preprocessing
    enhancedImg = adapthisteq(grayImg);
    denoisedImg = medfilt2(enhancedImg, [5 5]);
    
    % Adaptive thresholding
    binaryMask = imbinarize(denoisedImg, 'adaptive', 'Sensitivity', 0.6);
    
    % Morphological cleanup
    se = strel('disk', 3);
    cleanMask = imopen(binaryMask, se);
    filledMask = imfill(cleanMask, 'holes');
    
    % Edge refinement
    edges = edge(denoisedImg, 'Canny');
    finalMask = filledMask & ~edges;
    finalMask = bwareaopen(finalMask, 100);
    
    % Feature extraction
    stats = regionprops(finalMask, {'Area', 'Circularity', 'Solidity'});
    
    % Decision making
    if isempty(stats)
        tumorDetected = false;
        features = struct('Area', 0, 'Circularity', 0, 'Solidity', 0);
    else
        [~, idx] = max([stats.Area]);
        features.Area = stats(idx).Area;
        features.Circularity = stats(idx).Circularity;
        features.Solidity = stats(idx).Solidity;
        
        tumorDetected = (features.Area > params.areaThreshold) && ...
                       (features.Circularity > params.circularityThreshold) && ...
                       (features.Solidity > params.solidityThreshold);
    end
    
    % Display results
    boundary = bwperim(finalMask);
    overlayImg = imoverlay(inputImg, boundary, [1 0 0]);
    
    figure('Name', sprintf('%s Image %d - %s', imgType, imgNum, imgPath), ...
           'NumberTitle', 'off');
    
    subplot(2,2,1); imshow(inputImg); title('Original');
    subplot(2,2,2); imshow(denoisedImg); title('Preprocessed');
    subplot(2,2,3); imshow(finalMask); title('Segmentation');
    subplot(2,2,4); imshow(overlayImg); 
    title(sprintf('Result: %s\nArea: %.1f, Circ: %.2f, Solid: %.2f', ...
          string(tumorDetected), features.Area, ...
          features.Circularity, features.Solidity));
    
    % Print console output
    fprintf('Image %02d: ', imgNum);
    if tumorDetected
        fprintf('Tumor detected');
    else
        fprintf('No tumor');
    end
    fprintf(' (Area=%.1f, Circ=%.2f, Solid=%.2f)\n', ...
            features.Area, features.Circularity, features.Solidity);
end