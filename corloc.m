function [ratio, tp, gt] = corloc(bboxes, confidences, image_ids, VOCopts) 

addpath('VOCdevkit/VOC2007')


classes={...
        'aeroplane'
        'bicycle'
        'bird'
        'boat'
        'bottle'
        'bus'
        'car'
        'cat'
        'chair'
        'cow'
        'diningtable'
        'dog'
        'horse'
        'motorbike'
        'person'
        'pottedplant'
        'sheep'
        'sofa'
        'train'
        'tvmonitor'};

gt = zeros(numel(classes), 1) ;
tp = zeros(numel(classes), 1) ; 

for l = 1:numel(image_ids)
    anno = PASreadrecord(sprintf(VOCopts.annopath,image_ids{l}));
    scores = confidences{l};
    boxes = bboxes{l};

    
    i=3;
       [~, rank] = sort(-scores(i, :));
       best_box = boxes(rank, :);
       best_box = best_box(1, :);
       best_box = [best_box(2), best_box(1), best_box(4)-best_box(2), best_box(3)-best_box(1)] ; 
       
       flag = false;
       bestIOU = -1.0 ; 
       
       for j = 1:size(anno.objects, 2)
            if(strcmp(classes{i}, anno.objects(j).class) == 1)
                if ~flag
                    flag = true;
                    gt(i) = gt(i) + 1 ;
                end
                anno_box = [anno.objects(j).bbox(1:2), anno.objects(j).bbox(3)-anno.objects(j).bbox(1), anno.objects(j).bbox(4)-anno.objects(j).bbox(2)] ;
                iou = bboxOverlapRatio(anno_box, best_box, 'Union') ;
                if iou > bestIOU 
                    bestIOU = iou ; 
                end
            end
       end
       
       if bestIOU > 0.5 ; tp(i) = tp(i) + 1 ; end 
    
    if mod(l, 100) == 0; fprintf('%d/%d images complete\n', l, numel(db.names)) ; end 
end

ratio = tp ./ gt ; 
