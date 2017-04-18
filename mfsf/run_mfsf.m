function run_mfsf(path_in, name, nref, nframe, vis, pad, maxpix, hd)

    if (nargin < 5) vis = 0; end
    if (nargin < 6) pad = 4; end
    if (nargin < 7) maxpix = 200000; end
    if (nargin < 8) hd = 1; end

    fn = ['frame_%0' num2str(pad) 'd.tif'];

	path_res = ['./mfsf_output/' name]; 
	%[u,v,parmsOF,info] = runMFSF('path_in',path_in,'frname_frmt','frame_%03d.tif',...
	% 'nref', nref, 'sframe', 1, 'nframe', nframe, 'STDfl', 0, 'MaxPIXpyr', 20000, 'alpha', 20,...
	% 'flag_grad', 1);
	
	display(['MaxPIXpyr ' num2str(maxpix)]);
	[u,v,parmsOF,info] = runMFSF('path_in',path_in,'frname_frmt',fn,...
	 'nref', nref, 'sframe', 1, 'nframe', nframe, 'STDfl', 1, 'MaxPIXpyr', maxpix, 'alpha', 30,...
	 'flag_grad', 2);

	% Save the result:
	mkdir(path_res);
	if (hd == 1)
		save(fullfile(path_res,'result.mat'), 'u', 'v', 'parmsOF','info', '-v7.3');
	else 
		save(fullfile(path_res,'result.mat'), 'u', 'v', 'parmsOF','info');	

	if vis > 0
		path_figs = fullfile(path_res,'figures');
		tic; visualizeMFSF(path_in,u,v,parmsOF, 'path_figs',path_figs, 'Nrows_grid', 60,...
		    'file_mask_grid','in_test/mask_f89.png');
		fprintf('\nRuntime of MFSF algorithm: %g sec (%g sec per frame)\nRuntime of visualisation code: %g sec\n', ...
		    info.runtime,info.runtime/parmsOF.nframe,toc);
	end
end
