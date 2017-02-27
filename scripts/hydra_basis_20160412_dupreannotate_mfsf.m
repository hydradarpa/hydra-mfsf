display(['MFSF with annotated basis'])
nref = 100;
str = 1;
stk = 1;
nframe = 200;
path_in = '../hydra/video/20160412/stk_0001/';
name = sprintf('stack%04d_sframe%d_nref%d_nframe%d_annotatedbasis',stk, str, nref, nframe); 
run_mfsf_basis(path_in, name, nref, nframe);