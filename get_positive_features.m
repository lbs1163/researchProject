% Starter code prepared by James Hays for CS 143, Brown University
% This function should return all positive training examples (faces) from
% 36x36 images in 'train_path_pos'. Each face should be converted into a
% HoG template according to 'feature_params'. For improved performance, try
% mirroring or warping the positive training examples.

function features_pos = get_positive_features(VOCopts, cls, feature_params)
% 'train_path_pos' is a string. This directory contains 36x36 images of
%   faces
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.


% 'features_pos' is N by D matrix where N is the number of faces and D
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
features_pos=cell(1, num_images);
D = (feature_params.template_size / feature_params.hog_cell_size)^2 * feature_params.hog_dimension;
load ('SelectiveSearchVOC2007trainval.mat');
tic;
for i = 1:num_images
    % display progress
    if toc>1
        fprintf('%s: get positive feature: %d/%d\n',cls,i,num_images);
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
    
    if gt==1
        % extract features for image
        % compute and save features
        image = single(imread(sprintf(VOCopts.imgpath,image_files{i})));
        image_boxes = boxes{i};
        features_pos{i}=zeros(size(image_boxes, 1), D);
        for j=1:size(image_boxes, 1)
            template = imresize(image(image_boxes(j,1):image_boxes(j,3), image_boxes(j,2):image_boxes(j,4)), [72 72]);
            template_hog=vl_hog(template, feature_params.hog_cell_size);
            features_pos{i}(j, :)=reshape(template_hog, 1, []);
        end
    end
end

save('features_pos.mat', 'features_pos', '-v7.3');
