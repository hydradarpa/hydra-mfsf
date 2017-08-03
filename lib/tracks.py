"""Being things to do with particle tracks"""

def load_ground_truth(csv_file_name):
	# return a array of frame or a list of dict of tuples indexed by nueron id (x-int, y-int, rad-radius)

	cells = []
	rows = np.loadtxt(csv_file_name, usecols=[2,4,5,7], delimiter=',', dtype=float)
	max_frame = 0
	min_frame = 100
	for row in rows:
		frame = int(row[3])
		max_frame = max(max_frame, frame)
		min_frame = min(min_frame, frame)

	if min_frame != 0:
		print('csv file starting frame is not 0')

	cells = [dict() for _ in range(max_frame+1)]

	#File format
	#   0     1 2  3      4      5 6 7
	#           *         *      *   *
	# 102,61416,1,-1,201.38,496.75,0,0,0,82056,1,2.6533e+05,76.243,232,54,255,7423,76.198
	#           * neuron_id
	#                     * x
	#                            * y 
	#                                * frame

	for row in rows:
		neuron_id = int(row[0])
		x = int(Decimal(row[1]).to_integral_value(context=BasicContext))
		y = int(Decimal(row[2]).to_integral_value(context=BasicContext))
		frame = int(row[3])
		(cells[frame])[neuron_id] = [x,y]

	# load override list
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		rows = np.loadtxt(csv_file_name + '-override', usecols=[0,1,2,3], delimiter=',',dtype=int)

	for row in rows:
		neuron_id = int(row[0])
		x = int(Decimal(row[1]).to_integral_value(context=BasicContext))
		y = int(Decimal(row[2]).to_integral_value(context=BasicContext))
		frame = int(row[3])
		(cells[frame])[neuron_id] = [x,y]

	return [cells, max_frame]

def save_tracks_csv(fn_out, tracks):
	#Also save a csv file
	fh = open(fn_out, 'w')
	for n_id in tracks:
		for [x,y,frame] in tracks[n_id]:
			fh.write("%d,%d,%d,%d\n"%(n_id, frame, x, y))
	fh.close()

def load_tracks_csv(fn_in):
	cells = {}
	#Also save a csv file
	fh = open(fn_in, 'r')
	for line in fh:
		[n_id, frame, x, y] = [int(float(d)) for d in line.split(',')]
		if n_id in cells:
			cells[n_id].append([x,y,frame])
		else:
			cells[n_id] = [[x,y,frame]]
	fh.close()
	return cells

def filter_by_time(tracks, start, end):
	#Remove paths that don't survive for the entire specified range
	tn = tracks.copy()
	for n_id in tracks:
		if (tracks[n_id][0][2] > start) or (tracks[n_id][-1][2] < end):
			del tn[n_id]
	return tn 