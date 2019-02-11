function train_res_iden_MpRL3(num_predefined_classes, dropoutrate, num_real, num_gan, varargin)
% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------

% Load character dataset
imdb = load(strcat('./data/url_data_gan_24000.mat')) ;
imdb = imdb.imdb;

imdb.images.data(num_real+num_gan+1:end) = [];
imdb.images.label(num_real+num_gan+1:end) = [];
imdb.images.set(num_real+num_gan+1:end) = [];

imdb.images.set(:) = 1;

% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------
net = resnet52_iden_MpRL3(num_predefined_classes, dropoutrate);
net.conserveMemory = true;
net.meta.normalization.averageImage = reshape([105.6615,99.1308,97.9115],1,1,3);
% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------
opts.train.averageImage = net.meta.normalization.averageImage;
opts.train.batchSize = 64;
opts.train.continue = true;
opts.train.gpus = 1;
opts.train.prefetch = false ;
opts.train.nesterovUpdate = true ;
opts.train.expDir = './result/res52_iden_MpRL3';
% opts.train.derOutputs = {'objective', 0,'objective_multi_pseudo',1} ;
opts.train.derOutputs = {'objective',1,'objective_multi_pseudo',0} ;
%opts.train.gamma = 0.9;
opts.train.momentum = 0.9;
%opts.train.constraint = 100;
opts.train.learningRate = [0.1*ones(1,40),0.01*ones(1,10)] ;
opts.train.weightDecay = 0.0005;
opts.train.numEpochs = numel(opts.train.learningRate) ;
[opts, ~] = vl_argparse(opts.train, varargin) ;
% Call training function in MatConvNet
[net,info] = cnn_train_dag(net, imdb, @getBatch,opts) ;

% --------------------------------------------------------------------
function inputs = getBatch(imdb,batch,opts)
% --------------------------------------------------------------------
if(opts.epoch>20)  % after 20 epoch we start using pseudo label.
    opts.derOutputs = {'objective', 0,'objective_multi_pseudo',1} ;
end
im_url = imdb.images.data(batch) ;
im = vl_imreadjpeg(im_url,'Pack','Resize',[224,224],'Flip',...
    'CropLocation','random','CropSize',[0.85,1],...
    'Interpolation', 'bicubic','NumThreads',8);
labels = imdb.images.label(batch);

oim = bsxfun(@minus,im{1},opts.averageImage);
inputs = {'data',gpuArray(oim),'label',labels};
