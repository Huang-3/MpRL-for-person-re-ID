dataset_name = 'Market-1501-v15.09.15';
number_samples = 24000;

load(strcat('./data/url_data.mat'));
p = dir(strcat('/home/yan/datasets/',dataset_name,'/generated_54000_256/*.jpg'));

num = numel(imdb.images.data);
for i=1:number_samples
    url = strcat('/home/yan/datasets/',dataset_name,'/generated_54000_256/',p(i).name);
    imdb.images.data(num + i) =cellstr(url);
    imdb.images.label(num + i) = 0;
    imdb.images.set(num + i) = 1;
end

save(strcat('./data/url_data_gan_',num2str(number_samples),'.mat'),'imdb','-v7.3');