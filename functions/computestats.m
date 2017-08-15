function stats = computestats(est_tracks, real_tracks, assign)

	stats.lifetimes = [];
	stats.rms = [];
	stats.prop_gt = 0;
	stats.prop_est = 0;
	stats.minlens = [];
	stats.prop_lessthanthr = [];
	stats.matched = [];
	stats.thr = 6;

	%Extract the matching tracks
	[true_in_est,c] = find(assign);

	nRT = length(true_in_est);
	nET = size(assign,1)-nRT;

	stats.residual = cell(nRT,1);
	stats.times = cell(nRT,1);

	%Compare each matched track
	for idx = 1:nRT
		tr_gr = real_tracks(idx-1);
		tr_est = est_tracks(true_in_est(idx)-1);
		if length(tr_est) > 0
			[C,i1,i2] = intersect(tr_gr(:,1),tr_est(:,1));
			if length(C) > 0
				l_gr = size(tr_gr,1);
				l_est = size(tr_est,1);
				stats.lifetimes(end+1,:) = [l_gr, l_est];
				stats.minlens(end+1) = min(l_gr, l_est);
				p1 = tr_gr(i1,2:3);
				p2 = tr_est(i2,2:3);
				n = size(p1,1);
				rmse = sqrt(sum(sum((p1-p2).^2/n)));
				ds = sqrt(sum((p1-p2).^2,2));
				ts = i1;
				stats.times{idx} = ts;
				stats.residual{idx} = ds;
				stats.prop_lessthanthr(end+1) = sum(ds < stats.thr)/length(ds);
				stats.rms(end+1) = rmse;
				stats.matched(end+1) = 1;
			else
				stats.times{idx} = [];
				stats.residual{idx} = [];
				stats.rms(end+1) = 0;
				stats.prop_lessthanthr(end+1) = 0;
				stats.matched(end+1) = 0;
				stats.lifetimes(end+1,:) = 0;
				stats.minlens(end+1) = 0;
			end
		else 
			stats.times{idx} = [];
			stats.residual{idx} = [];
			stats.rms(end+1) = 0;
			stats.prop_lessthanthr(end+1) = 0;
			stats.matched(end+1) = 0;
			stats.lifetimes(end+1,:) = 0;
			stats.minlens(end+1) = 0;
		end
	end 

	stats.prop_gt = sum(stats.matched)/nRT;
	stats.prop_est = sum(stats.matched)/nET;

end