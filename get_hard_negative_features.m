% Starter code prepared by James Hays for CS 143, Brown University
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
function [features_pos, features_neg] = .... 
    get_hard_negative_features(features_pos, features_neg, w, b, threshold)


    pos_index_confidence = features_pos*w+b<threshold;
    neg_index_confidence = features_neg*w+b<threshold;
    features_neg = [features_neg(neg_index_confidence, :);features_pos(pos_index_confidence, :)];
    features_pos = [features_neg(~neg_index_confidence, :);features_pos(~pos_index_confidence, :)];

    

end




