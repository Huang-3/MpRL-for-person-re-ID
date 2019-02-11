
addpath(genpath('./code'));
addpath(genpath('./data'));
addpath(genpath('./result'));


num_predefined_classes = 751;
dropoutrate = 0.75;

num_real = 12936;
num_gan = 24000;

%% tran the baseline model 
% train_res_iden_baseline(num_predefined_classes, dropoutrate);

%% train 
% train_res_iden_MpRL3(num_predefined_classes, dropoutrate, num_real, num_gan);


%% 2stream model

% train_2stream_baseline(num_predefined_classes, dropoutrate);
train_2stream_MpRL3(num_predefined_classes, dropoutrate, num_real, num_gan);



