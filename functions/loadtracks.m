function tracks = loadtracks(fn_in)
	display(['Loading tracks from ' fn_in])
	tracks_csv = csvread(fn_in, 1, 0);
	%Build up structures with each set of tracks in it
	tracks = containers.Map('KeyType', 'double', 'ValueType', 'any');
	track_ids = unique(tracks_csv(:,1));
	for idx = 1:size(track_ids,1)
		tr = track_ids(idx);
		tracks(tr) = zeros(0,3);
	end
	for idx = 1:size(tracks_csv,1)
		t_idx = tracks_csv(idx, 1);
		pt_idx = tracks_csv(idx, 2:4);
		tracks(t_idx) = [tracks(t_idx); pt_idx];
	end	
end