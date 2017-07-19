function stats = tracking_performance(real_tracks_in, est_tracks_in, stats_out)

	CPDPATH = '~/matlab/CPD2';
	test = true;

	if nargin < 3
		stats_out = '';
		write_stats = false;
	else
		write_stats = true;
	end

	stats.lifetimes = [];
	stats.rms = [];
	stats.prop_gt = [];
	stats.prop_est = [];
	stats.minlens = [];

	addpath_recurse(CPDPATH);
	opt.method='nonrigid';  % use nonrigid registration
	opt.beta=.002;          % the width of Gaussian kernel (smoothness)
	opt.lambda=.003;        % regularization weight	
	opt.viz=0;              % show every iteration
	opt.outliers=0.4;       % noise weight
	opt.fgt=0;              % do not use FGT (default)
	opt.normalize=0;        % normalize to unit variance and zero mean before registering (default)
	opt.corresp=1;          % compute correspondence vector at the end of registration (not being estimated by default)
	opt.max_it=1000;        % max number of iterations
	opt.tol=1e-10;          % tolerance
	opt.beta = 10;			% make more rigid (default 2)

	%Load detection data from csv file
	%fn_in1 = real_tracks_in;
	%fn_in2 = est_tracks_in;

	%Test code defaults
	if test
		write_stats = false;
		fn_in1 = './tracks/20160412/20160412_dupreannotation_stk0001.csv';
		fn_in2 = './tracks/20160412/20160412_dupreannotation_stk0001.csv';
	end
	
	real_tracks = loadtracks(fn_in1);
	nRT = real_tracks.Count;
	est_tracks = loadtracks(fn_in2);
	nET = est_tracks.Count;

	%Determine number of frames...
	parpool(8);
	parfor refframe = 1:nF
		display(['Computing stats using ' num2str(refframe) ' as a reference'])
	
		%Make list of real_tracks that are in refframe
		real_track_refframe = [];
		for idx = 0:(nRT-1)
			t = real_tracks(idx);
			if any(t(:,1) == refframe)
				real_track_refframe(end+1) = idx;
			end
		end
	
		%Make list of est_tracks that are in refframe
		est_track_refframe = [];
		for idx = 0:(nRT-1)
			t = est_tracks(idx);
			if any(t(:,1) == refframe)
				est_track_refframe(end+1) = idx;
			end
		end
	
		%%%Find real tracks at refframe
		pts_real = zeros(0,3);
		for i = 1:length(real_track_refframe)
			idx = real_track_refframe(i);
			trk = real_tracks(idx);
			row = find(trk(:,1) == refframe);
			pt = [trk(row,2:3), idx];
			pts_real = [pts_real; pt];
		end
		Gr_ref = pts_real;
	
		%%%Find est tracks at refframe
		pts_est = zeros(0,3);
		for i = 1:length(est_track_refframe)
			idx = est_track_refframe(i);
			trk = est_tracks(idx);
			row = find(trk(:,1) == refframe);
			pt = [trk(row,2:3), idx];
			pts_est = [pts_est; pt];
		end
		D_ref = pts_est;
			
		%Transform such that Y corresponds to X(C,:)
		[Transform, C_ref] = cpd_register(Gr_ref(:,1:2), [D_ref(:,1:2); 1 1; 2 2; 3 3], opt);
		
		%Things that don't get matched go to 1...remove these points
		matched = C_ref~=1;
		Gr_ref_m = Gr_ref(C_ref,:);
		match_ref = [Gr_ref_m(matched,:), D_ref(matched,:)];
		
		%First column is ground truth, second is estimated tracks
		matched_tracknos = match_ref(:,[3,6]);
		
		nreal = size(Gr_ref,1);
		nest = size(D_ref,1);
		stats_ref = stattracks(real_tracks, est_tracks, matched_tracknos, nreal, nest);
		append_stats(stats_ref, stats);
	end
		
	%Save this data for plotting
	if write_stats
		save(stats_out, stats);
	end
end

function append_stats(stats_in, stats)
	
	stats.lifetimes = [stats.lifetimes; stats_in.lifetimes];
	stats.rms = [stats.rms stats_in.rms];
	stats.prop_gt = [stats.prop_gt stats_in.prop_gt];
	stats.prop_est = [stats.prop_est stats_in.prop_est];
	stats.minlens = [stats.minlens stats_in.minlens];

end