
function train_res_iden_baseline(num_predefined_classes, dropoutrate, varargin)
% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------

% Load character dataset
imdb = load(strcat('./data/url_data.mat'));
imdb = imdb.imdb;

imdb.images.set(:) = 1;

% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------
net = resnet52_iden_baseline(num_predefined_classes, dropoutrate);
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
opts.train.expDir = './result/res52_iden_baseline';
opts.train.derOutputs = {'objective', 1} ;
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
oim = bsxfun(@minus,im{1},opts.averageImage);
inputs = {'data',gpuArray(oim),'label',labels};