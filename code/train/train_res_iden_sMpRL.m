
function train_res_iden_sMpRL(num_predefined_classes, dropoutrate, num_real, num_gan, varargin)
% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------

% Load character dataset
imdb = load('./data/url_data_gan_24000.mat') ;
multipse_label = load('./data/sMpRL_24000');
multipse_label = multipse_label.multipse_label;
imdb = imdb.imdb;

multipse_label(:,num_gan+1:end) = [];
imdb.images.data(num_real+num_gan+1:end) = [];
imdb.images.label(num_real+num_gan+1:end) = [];
imdb.images.set(num_real+num_gan+1:end) = [];

imdb.images.set(:) = 1;

iden_num = size(multipse_label,1);
sparse_rate = 0;

k=1;
for i=1:numel(imdb.images.label)
    if imdb.images.label(i)==0
        if sparse_rate ~= 0
            sparse_label = multipse_label(:,k);
            sparse_label(sparse_label <= ceil(iden_num * sparse_rate) / iden_num) = 0;
            imdb.images.multipse_label{i} = sparse_label;
        else
            imdb.images.multipse_label{i} = multipse_label(:,k);
        end
        k=k+1;
    else
        imdb.images.multipse_label{i} = zeros(num_predefined_classes,1);
    end
end

% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------
net = resnet52_iden_sMpRL(num_predefined_classes, dropoutrate);
net.conserveMemory = true;
net.meta.normalization.averageImage = reshape([105.6615,99.1308,97.9115],1,1,3);
% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------
opts.train.averageImage = net.meta.normalization.averageImage;
opts.train.batchSize = 64;
opts.train.continue = false;
opts.train.gpus = 1;
opts.train.prefetch = false ;
opts.train.nesterovUpdate = true ;
opts.train.expDir = strcat('./result/res52_iden_sMpRL',num2str(num_gan));
opts.train.derOutputs = {'objective_multi_pseudo', 1} ;
%opts.train.gamma = 0.9;
opts.train.momentum = 0.9;
%opts.train.constraint = 100;
opts.train.learningRate = [0.1*ones(1,40),0.01*ones(1,10)] ;
opts.train.weightDecay = 0.0001;
opts.train.numEpochs = numel(opts.train.learningRate) ;
[opts, ~] = vl_argparse(opts.train, varargin) ;
% Call training function in MatConvNet
[net,info] = cnn_train_dag(net, imdb, @getBatch,opts) ;

% --------------------------------------------------------------------
function inputs = getBatch(imdb,batch,opts)
% --------------------------------------------------------------------
im_url = imdb.images.data(batch) ;
im = vl_imreadjpeg(im_url,'Pack','Resize',[224,224],'Flip',...
    'CropLocation','random','CropSize',[0.85,1],...
    'Interpolation', 'bicubic','NumThreads',8);
labels = imdb.images.label(batch);
multipse_label = imdb.images.multipse_label(batch);
multipse_label = cell2mat(multipse_label);
multipse_label = reshape(multipse_label,1,1,size(multipse_label,1),size(multipse_label,2));
oim = bsxfun(@minus,im{1},opts.averageImage);
inputs = {'data',gpuArray(oim),'label',labels,'multipse_label',gpuArray(multipse_label)};
