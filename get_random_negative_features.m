% Starter code prepared by James Hays for CS 143, Brown University
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale, because the positive training data is only available in
% grayscale. For best performance, you should sample random negative
% examples at multiple scales.

function features_neg = get_random_negative_features(VOCopts, cls, feature_params, num_samples)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.

% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray



image_files = textread(sprintf(VOCopts.imgsetpath,'trainval'),'%s');
num_images = length(image_files);
D = (feature_params.template_size / feature_params.hog_cell_size)^2 * feature_params.hog_dimension;
n_features_per_image = ceil(num_samples/num_images);
t_size = feature_params.template_size;
features_neg = zeros(n_features_per_image*num_images, D);
tic;
for i = 1:num_images
    % display progress
    if toc>1
        fprintf('%s: get random negative: %d/%d\n',cls,i,num_images);
        drawnow;
        tic;
    end
    
    % read annotation
    rec=PASreadrecord(sprintf(VOCopts.annopath,image_files{i}));
    
    % find objects of class and extract difficult flags for these objects
    clsinds=strmatch(cls,{rec.objects(:).class},'exact');
    diff=[rec.objects(clsinds).difficult];
    
    % assign ground truth class to image
    if isempty(clsinds)
        gt=-1;          % no objects of class
    elseif any(~diff)
        gt=1;           % at least one non-difficult object of class
    else
        gt=0;           % only difficult objects
    end
    
    if gt == -1
        % extract features for image
        % compute and save features
        img = single(imread(sprintf(VOCopts.imgpath,image_files{i})));scale = rand();
        min_scale = 2*max(feature_params.template_size/size(img, 1), feature_params.template_size/size(img, 2));
        image = imresize(img, max(min_scale, scale));
        [height, width, channel] = size(image);
        for j = 1:n_features_per_image
            offset_x = ceil(rand()*(width-t_size));
            offset_y = ceil(rand()*(height-t_size));
            
            part_image = image(offset_y:offset_y+t_size-1, offset_x:offset_x+t_size-1);
            features_neg((i-1)*n_features_per_image+j, :) = reshape(vl_hog(single(part_image), feature_params.hog_cell_size), [], 1);
        end
    end
end
save('features_rand_neg.mat', 'features_neg', '-v7.3');

