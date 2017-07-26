function stats = tracking_performance_munkres(real_tracks_in, est_tracks_in, stats_out)

	%Use the Munkres algorithm to associate set of tracks based on cost function between tracks

	test = false;

	if nargin < 3
		stats_out = '';
		write_stats = false;
	else
		write_stats = true;
	end

	%Load detection data from csv file
	%fn_in1 = real_tracks_in;
	%fn_in2 = est_tracks_in;

	nF = 99;

	%Test code defaults
	if test
		write_stats = false;
		real_tracks_in = './tracks/20160412/20160412_dupreannotation_stk0001.csv';
		est_tracks_in = './tracks/20160412/jerry_motion_corrected.csv';
	end
	
	real_tracks = loadtracks(real_tracks_in);
	nRT = real_tracks.Count;
	est_tracks = loadtracks(est_tracks_in);
	nET = est_tracks.Count

	%est_tracks.keys

	%Generate cost matrix
	%X = ground truth
	%Y = est tracks
	%Y_tilde = padded est tracks w dummy tracks

	%Pad Y with |X| dummy elements
	for idx = 0:(nRT-1)
		est_tracks(nET+idx) = [];
	end
	nET = nET + nRT;

	%For these tracks compute the cost matrix
	cost = zeros(nET, nRT);
	for i = 0:(nET-1)
		i;
		t1 = est_tracks(i);
		for j = 0:(nRT-1)
			t2 = real_tracks(j);
			cost(i+1,j+1) = trackdistance(t1, t2);
		end
	end

	%Run Munkres to assign tracks with one another
	%Takes around 10-15 minutes to run
	[assign, cost_min] = munkres(cost);
	stats = computestats(est_tracks, real_tracks, assign);

	%Save this data for plotting
	if write_stats
		save(stats_out, stats);
	end
end

function c = trackdistance(t1, t2, ep)
	%Compute the distance between two tracks with an unassigned/lost penalty of 10
	if (nargin < 3) ep = 10; end
	l1 = size(t1,1);
	l2 = size(t2,1);
	%Take care of special cases
	if l1 == 0 & l2 > 0
		c = ep*size(t2,1);
	elseif l1 > 0 & l2 == 0
		c = ep*size(t1,1);
	elseif l1 == 0 & l2 == 0
		c = 0;
	%Then the general case
	else 
		%Find overlapping times...
		[C,i1,i2] = intersect(t1(:,1),t2(:,1));
		matched = size(C,1);
		p1 = t1(i1,2:3);
		p2 = t2(i2,2:3);
		d = sqrt(sum((p1-p2).^2,2));
		c = sum(min(d,ep));
		%Non overlapping times get an 'ep' penalty
		c = c + ep*(l1-matched) + ep*(l2-matched);
	end
end