% function output = rand_diff_class(imdb,index,maxlabel)
% % uniform possible for every train image
%     output = randi(numel(imdb.images.reallabel));
%     while(imdb.images.reallabel(output) == imdb.images.reallabel(index) || imdb.images.set(output)~=1 ...
%             || (imdb.images.reallabel(output)<=maxlabel && imdb.images.reallabel(index)>maxlabel) ...
%             || (imdb.images.reallabel(output)>maxlabel && imdb.images.reallabel(index)<=maxlabel))
%         output = randi(numel(imdb.images.reallabel)); 
%     end
% end


function output = rand_diff_class(imdb,index)
% uniform possible for every train image
    output = randi(numel(imdb.images.label));
    while(imdb.images.label(output) == imdb.images.label(index) || imdb.images.set(output)~=1)
        output = randi(numel(imdb.images.label)); 
    end
end