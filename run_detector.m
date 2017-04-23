% Starter code prepared by James Hays for CS 143, Brown University
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
function [bboxes, confidences, image_ids] = .... 
    run_detector(VOCopts, w, b, feature_params)
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i. 
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.

% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your ale_stepinitial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression.

test_scenes = textread(sprintf(VOCopts.imgsetpath,'val'),'%s');

cell_size = feature_params.hog_cell_size;
num_cell_in_window = feature_params.template_size / cell_size;
D = (num_cell_in_window)^2 * feature_params.hog_dimension;

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = zeros(0,1);

scales = feature_params.scale_step.^linspace(0, feature_params.scale_size-1, feature_params.scale_size);


for si = 1:length(test_scenes)
      
    fprintf('Detecting car in %s\n', test_scenes{si})
    img = imread( sprintf(VOCopts.imgpath,test_scenes{si}));
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    cur_bboxes = zeros(0,4);
    cur_confidences = zeros(0,1);
    cur_image_ids = zeros(0,1);
    
    
    for x_scale = scales
        for y_scale = scales
            [height, width] = size(img);
            img_resize = imresize(img, [height * y_scale, width * x_scale]);
            [height, width] = size(img_resize);

            test_features = vl_hog(img_resize, cell_size); % n_cell_x * n_cell_y * 31

            n_window_x = floor((width-feature_params.template_size)/cell_size)+1;
            n_window_y = floor((height-feature_params.template_size)/cell_size)+1;

            descriptors = zeros(n_window_x*n_window_y, D); % n_window * D
            for i = 1:n_window_x
                for j = 1:n_window_y
                    %fprintf('%d %d %d %d\n', i, j, n_window_x, n_window_y)
                    if (j+num_cell_in_window-1 <= n_window_y && i+num_cell_in_window-1 <= n_window_x)
                        %fprintf('%d %d\n', i, j)
                        descriptors((i-1)*n_window_y+j, :) = ...
                            reshape(test_features(j:(j+num_cell_in_window-1), i:(i+num_cell_in_window-1), :), 1, []);
                    end
                end
            end

            scores = descriptors*w + b;
            indices = find(scores>feature_params.threshold);

            left = floor((indices-1)./n_window_y);
            top = mod(indices, n_window_y)-1;

            cur_scale_bboxes = [(left*cell_size+1)/x_scale, (top*cell_size+1)/y_scale, ...
                (left+num_cell_in_window)*cell_size/x_scale, (top+num_cell_in_window)*cell_size/y_scale];
            cur_scale_confidences = scores(indices);
            cur_scale_image_ids = repmat(test_scenes{si}, size(indices, 1), 1);

            cur_bboxes      = [cur_bboxes;      cur_scale_bboxes];
            cur_confidences = [cur_confidences; cur_scale_confidences];
            cur_image_ids   = [cur_image_ids;   cur_scale_image_ids];
        end
    end
       
    
    %non_max_supr_bbox can actually get somewhat slow with thousands of
    %initial detections. You could pre-filter the detections by confidence,
    %e.g. a detection with confidence -1.1 will probably never be
    %meaningful. You probably _don't_ want to threshold at 0.0, though. You
    %can get higher recall with a lower threshold. You don't need to modify
    %anything in non_max_supr_bbox, but you can.
    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));

    cur_confidences = cur_confidences(is_maximum,:);
    cur_bboxes      = cur_bboxes(     is_maximum,:);
    cur_image_ids   = cur_image_ids(  is_maximum,:);
 
    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];
end




