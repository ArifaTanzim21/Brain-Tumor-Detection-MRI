clc;
clear all;
close all;


input_yes_dir = 'C:\Users\HP\OneDrive\Desktop\1911021_BME424_Project\Deep Learning\dataset\yes';
input_no_dir = 'C:\Users\HP\OneDrive\Desktop\1911021_BME424_Project\Deep Learning\dataset\no';
output_yes_dir = 'C:\Users\HP\OneDrive\Desktop\1911021_BME424_Project\Deep Learning\preprocessed_dataset\yes\';
output_no_dir = 'C:\Users\HP\OneDrive\Desktop\1911021_BME424_Project\Deep Learning\preprocessed_dataset\no\';


if ~exist(output_yes_dir, 'dir')
    mkdir(output_yes_dir)
end
if ~exist(output_no_dir, 'dir')
    mkdir(output_no_dir)
end

% Process YES images (with tumor)
process_images(input_yes_dir, output_yes_dir);

% Process NO images (without tumor)
process_images(input_no_dir, output_no_dir);

% ------------------- Image Processing Function -------------------
function process_images(input_dir, output_dir)
    files = dir(fullfile(input_dir, '*.jpg'));
    
    for i = 1:length(files)
        % Read image
        img_path = fullfile(input_dir, files(i).name);
        img = imread(img_path);
        
        % Convert to grayscale if RGB
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        
        % Median Filtering
        img = medfilt2(img, [3 3]);
        
        % Adaptive Histogram Equalization
        img = adapthisteq(img, 'NumTiles', [8 8], 'ClipLimit', 0.01);
        
        % Skull Stripping
        threshold = graythresh(img);
        brain_mask = imbinarize(img, threshold);
        brain_mask = imfill(brain_mask, 'holes');
        brain_mask = bwareaopen(brain_mask, 100);       
        brain_mask = bwareafilt(brain_mask, 1);           
        
        % Apply brain mask
        img = double(img) .* brain_mask;
        
        % Normalize intensity to [0, 1]
        img = img / 255;
        
        % Resize to 64x64
        img = imresize(img, [64 64]);
        
        % Convert to uint8 for saving
        img_to_save = im2uint8(img);
        
        % Save processed image
        output_path = fullfile(output_dir, files(i).name);
        imwrite(img_to_save, output_path);
        
        fprintf('Processed: %s\n', files(i).name);
    end
end
