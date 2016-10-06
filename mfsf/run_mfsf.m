function run_mfsf(path_in, name, nref, nframe)

	path_res = ['./mfsf_output/' name]; 
	[u,v,parmsOF,info] = runMFSF('path_in',path_in,'frname_frmt','frame_%03d.tif',...
	 'nref', nref, 'sframe', 1, 'nframe', nframe, 'MaxPIXpyr', 20000);
	
	% Save the result:
	mkdir(path_res);
	save(fullfile(path_res,'result.mat'), 'u', 'v', 'parmsOF','info');
	
	path_figs = fullfile(path_res,'figures');
	tic; visualizeMFSF(path_in,u,v,parmsOF, 'path_figs',path_figs, 'Nrows_grid', 60,...
	    'file_mask_grid','in_test/mask_f89.png');
	fprintf('\nRuntime of MFSF algorithm: %g sec (%g sec per frame)\nRuntime of visualisation code: %g sec\n', ...
	    info.runtime,info.runtime/parmsOF.nframe,toc);