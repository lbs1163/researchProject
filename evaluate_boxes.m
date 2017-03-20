function [gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_boxes(VOCopts, bboxes, confidences, image_ids, feature_params)
% 'bboxes' is Nx4, N is the number of non-overlapping detections, and each
% row is [x_min, y_min, x_max, y_max]
% 'confidences' is the Nx1 (final cascade node) confidence of each
% detection.
% 'image_ids' is the Nx1 image names for each detection.

%This code is modified from the 2010 Pascal VOC toolkit.
%http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2010/index.html#devkit

if(~exist('draw', 'var'))
    draw = 1;
end

%this lists the ground truth bounding boxes for the test set.

gt_ids = []; gt_bboxes = [];
image_files = textread(sprintf(VOCopts.imgsetpath,'trainval'),'%s');
for i=1:length(image_files)
    ann=PASreadrecord(sprintf(VOCopts.annopath,image_files{i}));
    index_d = ([ann.objects.difficult] == zeros(1, length([ann.objects.difficult])));
    gt_ids = [gt_ids; ann.filename];
    gt_bboxes = [gt_bboxes; double(ann.object.bbox)];
end

gt_isclaimed = zeros(length(gt_ids),1);
npos = size(gt_ids,1); %total number of true positives(label).

% sort detections by decreasing confidence
[sc,si]=sort(-confidences);
image_ids=image_ids(si);
bboxes   =bboxes(si,:);

% assign detections to ground truth objects
nd=length(confidences);
tp=zeros(nd,1);
fp=zeros(nd,1);
duplicate_detections = zeros(nd,1);
num_corloc = 0;
tic;
for d=1:nd
    % display progress
    if toc>1
        fprintf('pr: compute: %d/%d\n',d,nd);
        drawnow;
        tic;
    end
    cur_gt_ids = strcmp(image_ids{d}+'.jpg', gt_ids); %will this be slow?

    bb = bboxes(d,:);
    ovmax=-inf;

    for j = find(cur_gt_ids')
        bbgt=gt_bboxes(j,:);
        bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
        iw=bi(3)-bi(1)+1;
        ih=bi(4)-bi(2)+1;
        if iw>0 && ih>0       
            % compute overlap as area of intersection / area of union
            ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
               (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
               iw*ih;
            ov=iw*ih/ua;
            if ov>ovmax %higher overlap than the previous best?
                ovmax=ov;
                jmax=j;
            end
            if ov>0.5
                num_corloc = num_corloc+1;
            end
        end
    end
    
    % assign detection as true positive/don't care/false positive
    if ovmax >= 0.3
        if ~gt_isclaimed(jmax)
            tp(d)=1;            % true positive
            gt_isclaimed(jmax)=true;
        else
            fp(d)=1;            % false positive (multiple detection)
            duplicate_detections(d) = 1;
        end
    else
        fp(d)=1;                    % false positive
    end
end

% compute cumulative precision/recall
cum_fp=cumsum(fp);
cum_tp=cumsum(tp);
rec=cum_tp/npos; % 얼굴들을 다 detect했는지
prec=cum_tp./(cum_fp+cum_tp); % detect 한 얼굴이 진짜 얼굴인
feature_params.hog_cell_size
ap=VOCap(rec,prec);

fprintf('CorLoc for %s: %.3f\n',cls, num_corloc/npos);

if draw
    % plot precision/recall
    figure(12)
    plot(rec,prec,'-');
    axis([0 1 0 1])
    grid;
    xlabel 'recall'
    ylabel 'precision'
    title(sprintf('Average Precision = %.3f',ap));
    toString = @(var) evalc(['disp(var)']);
    annotation('textbox',[0.2 0.2 0.4 0.4], 'String', toString(feature_params))
    set(12, 'Color', [.988, .988, .988])
    
    pause(0.1) %let's ui rendering catch up
    average_precision_image = frame2im(getframe(12));
    % getframe() is unreliable. Depending on the rendering settings, it will
    % grab foreground windows instead of the figure in question. It could also
    % return a partial image.
    imwrite(average_precision_image, 'visualizations/average_precision.png')
    
    figure(13)
    plot(cum_fp,rec,'-')
    axis([0 300 0 1])
    grid;
    xlabel 'False positives'
    ylabel 'Number of correct detections (recall)'
    title('This plot is meant to match Figure 6 in Viola Jones');
end

%% Re-sort return variables so that they are in the order of the input bboxes
reverse_map(si) = 1:nd;
tp = tp(reverse_map);
fp = fp(reverse_map);
duplicate_detections = duplicate_detections(reverse_map);



