%% cuda-8.0 + cudnn-5.1

addpath matlab;
addpath examples;
run matlab/vl_setupnn ;

vl_compilenn('enableGpu', true, ...
'cudaRoot', '/usr/local/cuda', ...  %put cuda path here 
'cudaMethod', 'nvcc', ...
'enableCudnn',true,... 
'cudnnroot','/usr/local/cuda'); % put cudnn dir path here, which includes two dirs, `include` and `lib64`

warning('off');
