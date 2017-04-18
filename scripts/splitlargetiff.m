fn_in = '../hydra/video/20170219/20170219_4_cnqx2_REAL_EGGESTION.tif';
base_out = '../hydra/video/20170219/';

t = TIFFStack(fn_in);

stksz = 250;
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
	%writeTiff(fn_out, a);
	mini = min(min(min(a)), mini);
	maxi = max(max(max(a)), maxi);
end

maxi = max(maxi);
mini = min(mini);

maxi = 1000;

for stk = 1:nstks
	f = sprintf('../hydra/video/20170219/stk_%04d/stk_%04d.tif', stk, stk);
	burst_tiff(f, mini, maxi);
end

%Run MFSF on stacks
nref = 1; 
nframe = 250;
parpool(4);
parfor idx = 1:nstks-1
	display(['MFSF for stack ' num2str(idx)])
	path_in = sprintf('../hydra/video/20170219/stk_%04d/frames/',idx);
	name = sprintf('20170219/stack%04d_nref%d_nframe%d',idx, nref, nframe); 
	run_mfsf(path_in, name, nref, nframe);
end

%Select reference frames from interframes 

%Perform clustering on frames

%Run DeepMatching image segmentation