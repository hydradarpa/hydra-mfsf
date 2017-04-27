function splitlargetiff(name, vid_dir, stksz)

	if (nargin < 3) stksz = 250; end

	%name = 'earth';
	%vid_dir = '../hydra/video/';
	%base_out = '../hydra/video/earth/';
	%fn_in = '../hydra/video/earth/earth.tif';
	%stksz = 50;

	%base_out = '../hydra/video/20170219/';
	base_out = [vid_dir '/' name];
	%fn_in = '../hydra/video/20170219/20170219_4_cnqx2_REAL_EGGESTION.tif';
	fn_in =  [base_out '/' name '.tif'];
	
	t = TIFFStack(fn_in);
	display(['Writing ' num2str(stksz) ' frames at a time'])
	st_frames = 1:stksz:size(t,3);
	
	%Get the global max and min in the video
	u16 = 2^16-1;
	maxi = 0;
	mini = u16;
	
	nstks = length(st_frames);	
	for stk = 1:nstks
		st = st_frames(stk);
		display(['Writing frames ' num2str(st)]);
		mkdir(sprintf('%s/stk_%04d', base_out,stk));
		fn_out = sprintf('%s/stk_%04d/stk_%04d.tif', base_out, stk, stk);
		a = t(:,:,st:(st+stksz-1));
		%If not exist then make the stack
		if exist(fn_out, 'file') ~= 2
			writeTiff(fn_out, a);
		else
			display('Already created')
		end
		mini = min(min(min(a)), mini);
		maxi = max(max(max(a)), maxi);
	end
	
	maxi = max(maxi);
	mini = min(mini);
	maxi = 1000;
	
	for stk = 1:nstks
		f = sprintf('%s/%s/stk_%04d/stk_%04d.tif', vid_dir, name, stk, stk);
		burst_tiff(f, mini, maxi);
	end
	
	%Run MFSF on stacks (forwards)
	nref = 1; 
	nframe = stksz;
	parpool(4);
	path_in = '';
	mname = '';
	parfor idx = 1:nstks-1
		display(['MFSF for stack ' num2str(idx)])
		path_in = sprintf('%s/stk_%04d/frames/',base_out, idx);
		mname = sprintf('%s/stack%04d_nref%d_nframe%d', name, idx, nref, nframe); 
		run_mfsf(path_in, mname, nref, nframe);
	end
	
	%Run MFSF on stacks (backwards)
	nref = stksz; 
	nframe = stksz;
	parpool(4);
	parfor idx = 1:nstks-1
		display(['MFSF for stack ' num2str(idx)])
		path_in = sprintf('%s/stk_%04d/frames/', base_out, idx);
		mname = sprintf('%s/stack%04d_nref%d_nframe%d',name, idx, nref, nframe); 
		run_mfsf(path_in, mname, nref, nframe);
	end
end