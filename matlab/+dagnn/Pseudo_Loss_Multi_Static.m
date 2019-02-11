classdef Pseudo_Loss_Multi_Static < dagnn.Loss

  properties
    rate = 0;
    valid_pseudo_num = 0;
  end
  
  methods
    function outputs = forward(obj, inputs, params)
      x = inputs{1};
      c = inputs{2};
      c = reshape(c,1,1,1,size(c,2)); 
      multipse_label = inputs{3};
      
      obj.valid_pseudo_num = length(find(multipse_label~=0))/length(find(c==0));

      ci = label_to_index(x, c);
      
      [Xmax,index] = max(x,[],3) ;
      ex = exp(bsxfun(@minus, x, Xmax)) ;
      t1 = Xmax + log(sum(ex,3)) - x(ci) ; 
      K = size(x,3);
      % label smooth for all
      part1 = t1;  % softmax
      part2 = (bsxfun(@times,sum(multipse_label,3),log(sum(ex,3))) - ...
          sum(bsxfun(@times,multipse_label,bsxfun(@minus,x,Xmax)),3)) * (2*K) / ((2*K-obj.valid_pseudo_num+1)*obj.valid_pseudo_num);  % label smooth
      
      t = (1-obj.rate) * part1 + obj.rate *part2;
      gan = find(c==0);
      t(gan) = part2(gan);  % remove the softmax part for generated image.
      outputs{1} = sum(t(:));
      
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      x = inputs{1};
      c = inputs{2};
      c = reshape(c,1,1,1,size(c,2)); 
      multipse_label = inputs{3};
      
      ci = label_to_index(x, c);

      K = size(x,3);
      Xmax = max(x,[],3) ;
      ex = exp(bsxfun(@minus, x, Xmax)) ;
      part1 = bsxfun(@rdivide, ex, sum(ex,3));  % Y = E/L   (softmax result)
      real = find(c~=0);
      part1(ci(real)) = part1(ci(real)) - 1 ;  % for evey gt, d(gt) = Y-1
      part2 = bsxfun(@minus,bsxfun(@rdivide, ex, sum(ex,3)), multipse_label * (2*K) / ((2*K-obj.valid_pseudo_num+1)*obj.valid_pseudo_num));
      y = (1-obj.rate) * part1 + obj.rate *part2;  % When we set opt.rate=0, it equals to softmax loss
      gan = find(c==0);
      y(:,:,:,gan) = part2(:,:,:,gan); % We set opts.gan=1
      derInputs{1} = y;
      derInputs{2} = [];
      derInputs{3} = [];
      derParams = {};
    end

    function obj = Loss(varargin)
      obj.load(varargin) ;
    end
    
  end
end
