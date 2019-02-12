if ~exist('./data')
    mkdir('./data');
end

if ~exist('./result')
    mkdir('./result');
end

addpath(genpath('./code'));
addpath(genpath('./data'));
addpath(genpath('./result'));

if ~exist('./data/url_data.mat','file')
    prepare_data;
end
if ~exist('./data/url_data_gan_24000.mat','file')
    prepare_gan_data;
end
if ~exist('./data/sMpRL_24000.mat','file')
    prepare_sMpRL_label4data;
end


num_predefined_classes = 751;
dropoutrate = 0.75;

num_real = 12936;
num_gan = 24000;

%% tran the baseline model 
train_res_iden_baseline(num_predefined_classes, dropoutrate);

%% train
train_res_iden_sMpRL(num_predefined_classes, dropoutrate, num_real, num_gan);
train_res_iden_MpRL2(num_predefined_classes, dropoutrate, num_real, num_gan);
train_res_iden_MpRL3(num_predefined_classes, dropoutrate, num_real, num_gan);




