function burst_tiff(fn_in, mini, maxi)
	%Test case:
	%fn_in = './video/20170202/20170202-gcamp-ecto-10hz-1bin4-1000fr.tif';
	%fn_in = './20170202-gcamp-ecto-10hz-1bin4-1000fr.tif';

	u8 = 2^8-1;
	u16 = 2^16-1;
	nrm = true;

	if (nargin < 2) mini = u16;	end
	if (nargin < 3)
		maxi = 0;
	else
		nrm = false;
	end

	[base, name, ext] = fileparts(fn_in);
	path_res = [base '/frames/']; 
	mkdir(path_res);

	[base, name, ext] = fileparts(fn_in);
	path_res_u8 = [base '/frames8/']; 
	mkdir(path_res_u8);

	ts = TIFFStack(fn_in);
	nF = size(ts, 3);

	for idx = 1:nF
		display(['Writing frame ' num2str(idx)]);
		%Read frame
		fr = ts(:,:,idx);
		if nrm == true
			mini = min(min(min(fr)), mini);
			maxi = max(max(max(fr)), maxi);
		end
		fn_out = sprintf('%s/frame_%04d.tif', path_res, idx);
		%Write frame 
		imwrite(fr, fn_out);
	end

	for idx = 1:nF
		display(['Writing normalized 8 bit frame ' num2str(idx)]);
		%Read frame
		fr = ts(:,:,idx);
		if nrm == false
			fr = min(maxi, fr);
		end
		f8 = uint8(u8*double((fr-mini))/double(maxi-mini));
		fn_out = sprintf('%s/frame_%04d.tif', path_res_u8, idx);
		%Write frame 
		imwrite(f8, fn_out);
	end

	%Run avconv to make a video from the 8 bit frames
	cmd = sprintf('avconv -i %s/frame_%%04d.tif -c:v libx264 -crf 20 -y %s/%s.mp4', path_res_u8, base, name);
	system(cmd);
	
	%avconv -i ./frames8//frame_%04d.tif -c:v libx264 -crf 20 -y ./stk_0001.mp4
