path_in = '../hydra/video/20160412/stk_0002/';
name = 'stack0002_nref100'; 
nref = 100; 
nframe = 250;

run_mfsf(path_in, name, nref, nframe)


path_in = '../hydra/video/20160412/stk_0003/';
name = 'stack0003_nref100'; 
nref = 100; 
nframe = 250;

run_mfsf(path_in, name, nref, nframe)

path_in = '../hydra/video/20160412/stk_0004/';
name = 'stack0004_nref100_nframe250'; 
nref = 100; 
nframe = 250;

run_mfsf(path_in, name, nref, nframe);

nref = 100; 
nframe = 250;
for idx = 5:20
	display(['MFSF for stack ' num2str(idx)])
	path_in = sprintf('../hydra/video/20160412/stk_%04d/',idx);
	name = sprintf('stack%04d_nref%d_nframe%d',idx, nref, nframe); 
	run_mfsf(path_in, name, nref, nframe);
end

%avconv -framerate 5 -i frame_%03d.png -c:v huffyuv -y output.avi