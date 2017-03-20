function demo()
addpath('data/VOCdevkit/VOCcode')

db = load('results/WSDDN-results.mat')
check_all = false;

for l = 1:numel(db.names)
    img = imread(fullfile('data/VOCdevkit/VOC2007/JPEGImages/', db.names{l}));
    anno = PASreadrecord(fullfile('data/VOCdevkit/VOC2007/Annotations/', strcat(db.names{l}(1:6),'.xml')));
    scores = db.scores{l};
    boxes = db.boxes{l};

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
    
    for i = 1:size(scores, 1)
       [~, rank] = sort(-scores(i, :));
       rank_score = scores(i, rank);
       rank_box = boxes(rank, :);
       rank_box = [rank_box(:, 2), rank_box(:, 1), rank_box(:, 4)-rank_box(:, 2), rank_box(:, 3)-rank_box(:,1)];
       imshow(img);
       flag = false;
       axis on;
       hold on;
       gtrec = cell(1, 100);
       bestIOU = -1.0 ;
       for j = 1:size(anno.objects, 2)
            cnt = 0;
            if(strcmp(classes{i}, anno.objects(j).class) == 1)
                flag = true;
                cnt = cnt+1;
                anno_box = [anno.objects(j).bbox(1), anno.objects(j).bbox(2), ...
                    anno.objects(j).bbox(3)-anno.objects(j).bbox(1), anno.objects(j).bbox(4)-anno.objects(j).bbox(2)] ;
                gtrec{cnt} = rectangle('Position', anno_box, 'EdgeColor', 'g', 'LineWidth', 3);
                iou = bboxOverlapRatio(rank_box(1, :), anno_box) ;
                if iou > bestIOU; bestIOU = iou; end
            end
       end
       ct = text(double(5), double(5), classes{i});
       set(ct, 'color', 'b', 'fontSize', 15);
       if flag || check_all
           for j = 1:1
              jrec = rectangle('Position', rank_box(j, :), 'EdgeColor', 'r', 'LineWidth', 3); 
              ht = text(double(rank_box(j, 1) + 1), double(rank_box(j, 2) + 8), strcat(num2str(rank_score(j))));
              set(ht, 'color', 'b', 'fontSize', 10);
              fprintf('%dth image, %dth class, %dth Boxes, Position: [%d %d] - Score: %3.3f, IOU: %3.3f\n', l, i, j, rank_box(j, 1), rank_box(j, 2), rank_score(j), bestIOU);
              pause;
              delete(jrec);
              delete(ht);
           end
       end
       for j = 1:cnt
            delete(gtrec{j});
       end
       hold off;
    end
end
