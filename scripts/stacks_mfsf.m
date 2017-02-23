stk = 1; 
nframes = [10, 25, 50, 100, 200];

%stack0001_sframe21_nref21_nframe10
for nframe = nframes
	sframes = 1:nframe:200;
	for idx = 1:length(sframes)
		nref = sframes(idx);
		display(['MFSF for stack ' num2str(idx)])
		path_in = '../hydra/video/20160412/stk_0001/';
		name = sprintf('stack%04d_sframe%d_nref%d_nframe%d',stk, nref, nref, nframe); 
		run_mfsf(path_in, name, nref, nframe);
	end
end