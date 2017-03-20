function mod_window = modify_windows()
%Initial selective search window calculated by Koen van de Sande,
%University of Amsterdam

% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);

% initialize VOC options
VOCinit;

%load initial selective search window
load ('SelectiveSearchVOC2007trainval.mat');

image_files = textread(sprintf(VOCopts.imgsetpath,'trainval'),'%s');
num_images = length(image_files);
tic;
for n=1:num_images
    if toc>1
        fprintf('modifying %d image window\n',n);
        drawnow;
        tic;
    end
    % read annotation
    rec=PASreadrecord(sprintf(VOCopts.annopath,image_files{n}));
    %Get image size
    image_w = rec.imgsize(1);
    image_h = rec.imgsize(2);
    
    box=boxes{n};
    index_h=box(:, 1)<=image_h*0.04 | box(:, 3)>=(image_h-image_h*0.04);
    new_box=box(~index_h, :);
    index_w=new_box(:, 2)<=image_w*0.04 | new_box(:, 4)>=(image_w-image_w*0.04);
    new_box=new_box(~index_w, :);
    index_size=new_box(:, 3)-new_box(:, 1)<20 | new_box(:, 4)-new_box(:, 2)<20;
    new_box=new_box(~index_size, :);
    boxes{n}=new_box;
end

save ('SelectiveSearchVOC2007trainval.mat', 'images', 'boxes', '-v7.3');