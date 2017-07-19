function stats = stattracks(real_tracks, est_tracks, matched_tracknos, nreal, nest)
	%%Compute statistics comparing the two tracks
	%%%Lifetimes of the matched tracks
	%%%RMS of displacements of the matched tracks (when they both exist)
	%%%Proportion in ground truth, proportion in estimated

	lifetimes = zeros(0,2);
	rms = [];
	minlens = [];

	%For each matched track
	for idx = 1:size(matched_tracknos, 1)
		match = matched_tracknos(idx,:);
		tr_gr = real_tracks(match(1));
		tr_est = est_tracks(match(2));
		l_gr = size(tr_gr,1);
		l_est = size(tr_est,1);
		lifetimes(end+1,:) = [l_gr, l_est];
		minlen = min(l_gr, l_est);
		res = tr_gr(1:minlen,2:3)-tr_est(1:minlen,2:3);
		res2 = res.*res;
		rmse = sqrt(sum(sum(res2))/minlen);
		rms(end+1) = rmse;
		minlens(end+1) = minlen;
	end

	prop_gt = length(matched_tracknos)/nreal;
	prop_est = length(matched_tracknos)/nest;

	stats.lifetimes = lifetimes;
	stats.rms = rms;
	stats.prop_gt = prop_gt;
	stats.prop_est = prop_est;
	stats.minlens = minlens;

end

function tr_out = extract_contig(tr_in, ref)
	%Extract the contiguous region of a track that contains frame 'ref'
	%Maybe not needed...?
	tr_out = 1;


end