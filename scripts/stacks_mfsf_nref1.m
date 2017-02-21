nref = 1; 
nframe = 250;
parfor idx = 1:20
	display(['MFSF for stack ' num2str(idx)])
	path_in = sprintf('../hydra/video/20160412/stk_%04d/',idx);
	name = sprintf('stack%04d_nref%d_nframe%d',idx, nref, nframe); 
	run_mfsf(path_in, name, nref, nframe);
end
