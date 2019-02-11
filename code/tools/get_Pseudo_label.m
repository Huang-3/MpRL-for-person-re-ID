function f = get_Pseudo_label(net,oim,im_mean,inputname,outputname)
im = bsxfun(@minus,single(oim),im_mean);
net.vars(net.getVarIndex(outputname)).precious = true;
net.eval({inputname,gpuArray(im)}) ;
x = gather(net.vars(net.getVarIndex(outputname)).value);
f = vl_nnsoftmax(x);
end