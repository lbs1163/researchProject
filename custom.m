classifier = load('classifier.mat');

data_path = 'VOCdevkit/VOC2007'; %change if you want to work with a network copy
%custom_path = fullfile(data_path, 'custom'); %Positive training examples. 36x36 head crops
feature_params.threshold = 1.0;

[bboxes, confidences, image_ids] = run_detector(custom_path, classifier.w, classifier.b, feature_params);

visualize_detections_by_image_no_gt(bboxes, confidences, image_ids, custom_path);