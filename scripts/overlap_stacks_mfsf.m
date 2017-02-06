%Setup a stack with overlap (we'll call this mapping, instead of stitching)
nref = 250; 
nframe = 500;

%Make directory for tmp tiff files (if doesn't exist)
path_in = '../hydra/video/tmp/';
mkdir(path_in);

for idx = 1:7
	display(['MFSF for stack ' num2str(idx)])
	%Copy tiff files into tmp folder
	p1 = sprintf('../hydra/video/20160412/stk_%04d/',idx);
	p2 = sprintf('../hydra/video/20160412/stk_%04d/',idx+1);
	copyfile([p1 '/frame_*.tif'], path_in)
	files = dir([p2 '/frame_*.tif']);
	nF = size(files, 1);
	for j = 1:nF
		fn = files(j).name;
		fn_out = [path_in '/frame_' num2str(j+nref) '.tif'];
		%Get file number...
		copyfile([p2 '/' fn], fn_out)
	end

	path_out = sprintf('stack%04d_stack%04d',idx, idx+1); 
	run_mfsf(path_in, path_out, nref, nframe);
end