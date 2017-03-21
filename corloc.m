function [ratio, tp, gt] = corloc(bboxes, confidences, image_ids)

class='car';

gt = 0;
tp = 0;
image_names = unique(image_ids, 'rows');

for l = 1:size(image_names)
    anno = PASreadrecord(fullfile('VOCdevkit/VOC2007/Annotations/', strcat(image_names(l, :),'.xml')));
    idx = find(ismember(image_ids, image_names(l, :), 'rows'));
    confidences_idx = confidences(idx, :);
    boxes = bboxes(idx, :);
    
    [~, rank] = sort(-confidences_idx);
    best_box = boxes(rank, :);
    best_box = best_box(1, :);
    best_box = [best_box(2), best_box(1), best_box(4)-best_box(2), best_box(3)-best_box(1)] ;
    
    flag = false;
    bestIOU = -1.0 ;
    
    for j = 1:size(anno.objects, 2)
        if(strcmp(class, anno.objects(j).class) == 1)
            if ~flag
                flag = true;
                gt = gt + 1 ;
            end
            anno_box = [anno.objects(j).bbox(1:2), anno.objects(j).bbox(3)-anno.objects(j).bbox(1), anno.objects(j).bbox(4)-anno.objects(j).bbox(2)] ;
            iou = bboxOverlapRatio(anno_box, best_box, 'Union') ;
            if iou > bestIOU
                bestIOU = iou ;
            end
        end
    end
    
    if bestIOU > 0.5 ; tp = tp + 1 ; end
    
    if mod(l, 100) == 0; fprintf('%d/%d images complete\n', l, size(image_names, 1)) ; end
end

ratio = tp ./ gt ;
