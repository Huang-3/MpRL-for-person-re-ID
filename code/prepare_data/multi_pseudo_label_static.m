load('./data/url_data_gan_48000.mat');

path = 'res52_drop0.75_baseline';
netStruct = load(strcat('./result/', path, '/net-epoch-50.mat'));
net = dagnn.DagNN.loadobj(netStruct.net);

clear netStruct;
net.mode = 'test';
net.move('gpu') ;

net.conserveMemory = true;
im_mean = net.meta(1).normalization.averageImage;

idx = find(imdb.images.label==0);
prediction = [];

for i = idx(1):200:numel(imdb.images.label)
    disp(i);
    oim = [];
    str=[];
    for j=1:min(200,numel(imdb.images.label)-i+1)
        str = imdb.images.data{i+j-1};
        imt = imresize(imread(str),[224,224]);
        oim = cat(4,oim,imt);
    end
    temp = get_Pseudo_label(net,oim,im_mean,'data','prediction');
    prediction = cat(4,prediction,temp);
end
prediction = reshape(prediction,size(prediction,3),[]);
% save(fullfile('./multi_pseudo_labelgan_12000.mat'),'prediction','-v7.3');

iden_num = size(prediction,1);
label = 1/iden_num:1/iden_num:1;
multipse_label = zeros(size(prediction));
for i=1:size(prediction,2)
    x = prediction(:,i);
    [~, idx] = sort(x);
    multipse_label(:,i) = label(idx)';
    i
end
save(fullfile('./data/multi_pseudo_gan_48000.mat'),'multipse_label','prediction', '-v7.3');