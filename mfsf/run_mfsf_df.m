function run_mfsf_df(path_in, name, sframe, nref, nframe)

	path_res = ['./register/' name '/mfsf/ref_' num2str(nref)]; 
	[u,v,parmsOF,info] = runMFSF('path_in',path_in,'frname_frmt','frame_%03d.tif',...
	 'nref', nref, 'sframe', sframe, 'nframe', nframe, 'MaxPIXpyr', 20000);
	
	% Save the result:
	mkdir(path_res);
	save(fullfile(path_res,'result.mat'), 'u', 'v', 'parmsOF','info');
	
	path_figs = fullfile(path_res,'figures');
	tic; visualizeMFSF(path_in,u,v,parmsOF, 'path_figs',path_figs, 'Nrows_grid', 60,...
	    'file_mask_grid','in_test/mask_f89.png');
	fprintf('\nRuntime of MFSF algorithm: %g sec (%g sec per frame)\nRuntime of visualisation code: %g sec\n', ...
	    info.runtime,info.runtime/parmsOF.nframe,toc);

end