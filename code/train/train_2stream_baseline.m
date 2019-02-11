
function train_2stream_baseline(num_predefined_classes, dropoutrate, varargin)
% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------
% Load character dataset
imdb = load(strcat('./data/url_data.mat')) ;
imdb = imdb.imdb;
imdb.images.set(:) = 1;
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------
net = resnet52_2stream_baseline(num_predefined_classes, dropoutrate); 
net.conserveMemory = true;             
net.meta.normalization.averageImage = reshape([105.6615,99.1308,97.9115],1,1,3);
% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------
opts.train.averageImage = net.meta.normalization.averageImage;
opts.train.batchSize = 32;
opts.train.continue = true;
opts.train.gpus = 1;
opts.train.nesterovUpdate = true;
opts.train.prefetch = false ;
opts.train.expDir = './result/train_2stream_baseline';
opts.train.derOutputs = {'objective', 0.5,'objective_2', 0.5,'objective_final', 1} ;
opts.train.learningRate = [0.1*ones(1,40),0.01*ones(1,20)] ;                                                                                                                                                                                                                                                                                                                                                                                    
opts.train.weightDecay = 0.0001;
opts.train.numEpochs = numel(opts.train.learningRate) ;
[opts, ~] = vl_argparse(opts.train, varargin) ;
% Call training function in MatConvNet
[net,info] = cnn_train_dag(net, imdb, @getBatch,opts) ;

% --------------------------------------------------------------------
function inputs = getBatch(imdb, batch,opts)
% --------------------------------------------------------------------
im1_url = imdb.images.data(batch) ; 
label1 = imdb.images.label(:,batch) ;
batchsize = numel(batch);
% every epoch we will add negative pairs until 1:4
dividor = 2;
dividor = min(5,dividor*power(1.01,opts.epoch));
half = round(batchsize/dividor);

label_f = cat(1,ones(half,1,'single'),ones(batchsize-half,1,'single')*2);

batch2 = zeros(batchsize,1);
for i=1:batchsize
    if(i<=half)
        batch2(i) = rand_same_class(imdb, batch(i));
    else
        batch2(i) = rand_diff_class(imdb, batch(i));
    end
end
im2_url = imdb.images.data(batch2) ; 
im1 = vl_imreadjpeg(im1_url,'Pack','Resize',[224,224],'Flip',...
    'CropLocation','random','CropSize',[0.85,1],'interpolation','bicubic','NumThreads',8);
im2 = vl_imreadjpeg(im2_url,'Pack','Resize',[224,224],'Flip',...
    'CropLocation','random','CropSize',[0.85,1],'interpolation','bicubic','NumThreads',8);
label2 = imdb.images.label(:,batch2) ;

oim1 = bsxfun(@minus,im1{1},opts.averageImage); 
oim2 = bsxfun(@minus,im2{1},opts.averageImage);
label_f(label1==0) = 0;
inputs = {'data',gpuArray(oim1),'data_2',gpuArray(oim2),'label',label1,'label_2',label2,'label_f',label_f};

