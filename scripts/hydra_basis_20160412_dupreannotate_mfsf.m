display(['MFSF with annotated basis'])
nref = 100;
str = 1;
stk = 1;
nframe = 200;
vis = 0;
pad = 4;
bas_file = './analysis/20160412_dupreannotation_stk0001.csv';
maxpix = 200000;
path_in = '../hydra/video/20160412/stk_0001/';
name = sprintf('stack%04d_sframe%d_nref%d_nframe%d_annotatedbasis',stk, str, nref, nframe); 
run_mfsf_basis(path_in, name, nref, nframe, bas_file, vis, pad, maxpix);