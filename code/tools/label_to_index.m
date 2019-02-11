function [ ci ] = label_to_index( x, c )
%LABEL_TO_INDEX Summary of this function goes here
%   Detailed explanation goes here

    inputSize = [size(x,1) size(x,2) size(x,3) size(x,4)];
    labelSize = [size(c,1) size(c,2) size(c,3) size(c,4)];

    numPixelsPerImage = prod(inputSize(1:2)) ;
    numPixels = numPixelsPerImage * inputSize(4) ;
    imageVolume = numPixelsPerImage * inputSize(3) ;

    n = reshape(0:numPixels-1,labelSize) ;
    offset = 1 + mod(n, numPixelsPerImage) + ...
        imageVolume * fix(n / numPixelsPerImage) ;
    ci = offset + numPixelsPerImage * max(c - 1,0) ;

end

