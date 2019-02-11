% In this file, we densely extract the feature
% We extract feature from 256,256 image and mirrored image.
% -----Please change the file path in this script. ----
clear;
path = 'train_2stream_baseline';
netStruct = load(strcat('./result/',path,'/net-epoch-60.mat'));
if ~exist(strcat('./test/',path))
    mkdir(strcat('./test/',path));
end
dataset_name = 'Market-1501-v15.09.15';
%--------add L2-norm
net = dagnn.DagNN.loadobj(netStruct.net);
net.addLayer('lrn_test',dagnn.LRN('param',[4096,0,1,0.5]),{'pool5'},{'pool5n'},{});
clear netStruct;
net.mode = 'test';
net.move('gpu') ;
net.conserveMemory = true;
im_mean = net.meta(1).normalization.averageImage;
%im_mean = imresize(im_mean,[224,224]);
p = dir(strcat('/home/yan/datasets/',dataset_name,'/bounding_box_test/*.jpg'));
ff = [];
%------------------------------
for i = 1:200:numel(p)
    disp(i);
    oim = [];
    str=[];
    for j=1:min(200,numel(p)-i+1)
        str = strcat('/home/yan/datasets/',dataset_name,'/bounding_box_test/',p(i+j-1).name);
        imt = imresize(imread(str),[256,256]);
        oim = cat(4,oim,imt);
    end
    f = getFeature2(net,oim,im_mean,'data','pool5n');
    f = sum(sum(f,1),2);
    f2 = getFeature2(net,fliplr(oim),im_mean,'data','pool5n');
    f2 = sum(sum(f2,1),2);
    f = f+f2;
    size4 = size(f,4);
    f = reshape(f,[],size4)';
    s = sqrt(sum(f.^2,2));
    dim = size(f,2);
    s = repmat(s,1,dim);
    f = f./s;
    ff = cat(1,ff,f);
end
save(strcat('./test/',path,'/resnet_gallery.mat'),'ff','-v7.3');
%}

%---------query
p = dir(strcat('/home/yan/datasets/',dataset_name,'/query/*.jpg'));
ff = [];
for i = 1:200:numel(p)
    disp(i);
    oim = [];
    str=[];
    for j=1:min(200,numel(p)-i+1)
        str = strcat('/home/yan/datasets/',dataset_name,'/query/',p(i+j-1).name);
        imt = imresize(imread(str),[256,256]);
        oim = cat(4,oim,imt);
    end
    f = getFeature2(net,oim,im_mean,'data','pool5n');
    f = sum(sum(f,1),2);
    f2 = getFeature2(net,fliplr(oim),im_mean,'data','pool5n');
    f2 = sum(sum(f2,1),2);
    f = f+f2;
    size4 = size(f,4);
    f = reshape(f,[],size4)';
    s = sqrt(sum(f.^2,2));
    dim = size(f,2);
    s = repmat(s,1,dim);
    f = f./s;
    ff = cat(1,ff,f);
end
save(strcat('./test/',path,'/resnet_query.mat'),'ff','-v7.3');