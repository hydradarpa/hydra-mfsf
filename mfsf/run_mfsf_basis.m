function run_mfsf_basis(path_in, name, nref, nframe, bas_file, vis, pad)

    if (nargin < 6) vis = 0; end
    if (nargin < 7) pad = 3; end
    if (nargin < 5) bas_file = './analysis/20160412_dupreannotation_stk0001.csv'; end

    fn = ['frame_%0' num2str(pad) 'd.tif'];

	path_res = ['./mfsf_output/' name]; 
	%[u,v,parmsOF,info] = runMFSF('path_in',path_in,'frname_frmt','frame_%03d.tif',...
	% 'nref', nref, 'sframe', 1, 'nframe', nframe, 'STDfl', 0, 'MaxPIXpyr', 20000, 'alpha', 20,...
	% 'flag_grad', 1);

	%Prepare basis. From the help file:
	%bas: parameter specifying the basis: it can be an array or a string:
    %- if bas is an array it should be 2nframes x R, where R is the rank of the basis. In this case, this
    %array is the trajectory basis that you want to use. bas should be orthonormal, i.e. bas'*bas
    %should equal the identity matrix. Every column of bas should contain a
    %different basis element in the format [u(1) ... u(nframe) v(1) ...
    %v(nframe)]^T, where u and v are the horizontal and vertical components
    %of the displacement respectively. This means that the first nframes
    %rows of bas correspond to the horizontal displacement and the last
    %nfrmaes rows to the vertical displacement.

    %Load trajectories from path file
    tracks = load_tracks_csv(bas_file);
    %Make sure tracks are same length as video
    display('Truncating tracks to be within specified range')
    tracks = truncate_tracks(tracks, 0, nframe-1);
    tracksx = reshape(tracks(:,3), nframe, []);
    tracksy = reshape(tracks(:,4), nframe, []);

    tracksx = reshape(trks(:,3), nframe, []);
    tracksy = reshape(trks(:,4), nframe, []);
    bas = [tracksx; tracksy];
    display('Orthongalizing tracks')
    bas = orth(bas);

    %Format of trajectory file:
    %-xml?
    %-pkl
    %-mat file
	
    display('Running MFSF')
	[u,v,parmsOF,info] = runMFSF('path_in',path_in,'frname_frmt',fn,...
	 'nref', nref, 'sframe', 1, 'nframe', nframe, 'STDfl', 1, 'MaxPIXpyr', 20000, 'alpha', 30,...
	 'flag_grad', 2, 'bas', bas);

	% Save the result:
	mkdir(path_res);
	save(fullfile(path_res,'result.mat'), 'u', 'v', 'parmsOF','info', '-v7.3');
	
	if vis > 0
		path_figs = fullfile(path_res,'figures');
		tic; visualizeMFSF(path_in,u,v,parmsOF, 'path_figs',path_figs, 'Nrows_grid', 60,...
		    'file_mask_grid','in_test/mask_f89.png');
		fprintf('\nRuntime of MFSF algorithm: %g sec (%g sec per frame)\nRuntime of visualisation code: %g sec\n', ...
		    info.runtime,info.runtime/parmsOF.nframe,toc);
	end
end

function tracks = load_tracks_csv(fn_in)
	fh = fopen(fn_in, 'r');
	formatSpec = '%d,%d,%d,%d\n';
	tracks = fscanf(fh, formatSpec, [4 Inf]);
	tracks = tracks';
end

function trks = truncate_tracks(tracks, srt, nd)
	trks = [];
	for idx = 1:length(tracks)
		frm = tracks(idx,2);
		if frm >= srt && frm <= nd
			trks = [trks(:,:); tracks(idx,:)];
		end
	end
end