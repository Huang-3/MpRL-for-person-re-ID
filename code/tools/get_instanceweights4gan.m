function [ instanceWeights ] = get_instanceweights4gan(labels, weight)
%GET_INSTANCEWEIGHTS Summary of this function goes here
%   Detailed explanation goes here
  instanceWeights = ones(size(labels));
  gan = (labels==0);
  instanceWeights(gan) = weight;
  instanceWeights = reshape(instanceWeights,1,1,1,[]);

end

