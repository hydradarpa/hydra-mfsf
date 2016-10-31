%path_in = '../hydra/video/johntest_brightcontrast_short/';
%name = 'johntest_nref100'; 
%nref = 100; 
%nframe = 199;

%run_mfsf(path_in, name, nref, nframe)

for nref = 10:10:190
	path_in = '../hydra/video/johntest_brightcontrast_short/';
	name = ['johntest_nref' num2str(nref)]; 
	%nref = 10; 
	nframe = 199;

	run_mfsf(path_in, name, nref, nframe);
end

path_in = '../hydra/video/johntest_brightcontrast_short/';
name = 'johntest_flag_grad_1_nref100'; 
nref = 100; 
nframe = 199;

run_mfsf(path_in, name, nref, nframe);

path_in = '../hydra/video/johntest_brightcontrast_short/';
name = 'johntest_flag_STD_0_nref100'; 
nref = 100; 
nframe = 199;

run_mfsf(path_in, name, nref, nframe);


path_in = '../hydra/video/johntest_brightcontrast_short/';
name = 'johntest_flag_STD_0_alpha_10_nref100'; 
nref = 100; 
nframe = 199;

run_mfsf(path_in, name, nref, nframe);

path_in = '../hydra/video/johntest_brightcontrast_short/';
name = 'johntest_flag_STD_0_alpha_20_nref100'; 
nref = 100; 
nframe = 199;

run_mfsf(path_in, name, nref, nframe);

path_in = '../hydra/video/johntest_brightcontrast_short/';
name = 'johntest_flag_STD_0_alpha_20_flag_grad_1_nref100'; 
nref = 100; 
nframe = 199;

run_mfsf(path_in, name, nref, nframe);