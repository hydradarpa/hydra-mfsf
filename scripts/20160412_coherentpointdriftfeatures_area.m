clear all; close all; clc;
addpath_recurse('~/matlab/CPD2');

% Init full set of options %%%%%%%%%%
opt.method='nonrigid'; % use nonrigid registration
opt.beta=.002;            % the width of Gaussian kernel (smoothness)
opt.lambda=.003;          % regularization weight

opt.viz=1;              % show every iteration
opt.outliers=0.4;       % noise weight
opt.fgt=0;              % do not use FGT (default)
opt.normalize=0;        % normalize to unit variance and zero mean before registering (default)
opt.corresp=1;          % compute correspondence vector at the end of registration (not being estimated by default)

opt.max_it=1000;         % max number of iterations
opt.tol=1e-10;          % tolerance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

name = 'CPD_area';
refframe = 100;
nF = 200;

%Area may not work actually... 3D Registration's projection onto xy plane may not respect 
%coherence

%Load detection data from csv file
fn_in = './tracks/20160412/20160412_stk_0001_detections.csv';
%ROI,X,Y,T,Contour,Max. Feret diam.,Elongation ratio,Avg. ch 0
detections = csvread(fn_in, 1, 0);

matches = cell(nF,1);

t0 = detections(:,4)==refframe;
data0 = detections(t0,:);
X = [data0(:,2:3), data0(:,5)];

%for frame = 1:nF

parpool(6);
parfor frame = 1:nF
	if frame ~= refframe
		t1 = detections(:,4)==frame;
		data1 = detections(t1,:);	
		Y = [data1(:,2:3), data1(:,5)];
		%such that Y corresponds to X(C,:)
		[Transform, C] = cpd_register(X, Y, opt);
		matched = C~=1;
		Xm = X(C,:);
		d = [Xm(matched,:), Y(matched,:)];
	else
		d = [X,X];
	end
	matches{frame} = d;
end

%save(['./tracks/20160412/' name '.mat'], 'matches', 'refframe', 'nF', 'opt');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Once saved then compare the error based on matches with the true tracks%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%name = 'CPD_default';
%load(['./tracks/20160412/' name '.mat'], 'matches', 'refframe', 'nF', 'opt');

%Load ground truth
fn_in = './tracks/20160412/20160412_dupreannotation_stk0001.csv';
%Track #, time, X, Y
truth = csvread(fn_in, 1, 0);

%Load detection data from csv file
fn_in = './tracks/20160412/20160412_stk_0001_detections.csv';
%ROI,X,Y,T,Contour,Max. Feret diam.,Elongation ratio,Avg. ch 0
detections = csvread(fn_in, 1, 0);

%%%Find points at refframe
idx_ref = truth(truth(:,2) == refframe,1);
pts_ref = zeros(0,2);
for idx = 1:length(idx_ref)
	pt = truth(truth(:,1) == idx & truth(:,2) == refframe,3:4);
	pts_ref = [pts_ref; pt];
end

%Match ground truth in refframe to detections we find in refframe
Gr_ref = pts_ref;
t1 = detections(:,4)==refframe;
data1 = detections(t1,:);	
D_ref = data1(:,2:3);
%Transform such that Y corresponds to X(C,:)
opt.viz = 0;
opt.max_it = 1000;
[Transform, C_ref] = cpd_register(Gr_ref, D_ref, opt);
matched = C_ref~=1;
Gr_ref_m = Gr_ref(C_ref,:);
match_ref = [Gr_ref_m(matched,:), D_ref(matched,:)];

prop_D_gr = zeros(nF, 1);
prop_gr_D = zeros(nF, 1);

for frame = 1:nF
	display(['Matching for frame' num2str(frame)]);
	%Find correspondences between grouth truth refframe and ground truth current frame
	idx_cfr = truth(truth(:,2) == frame,1);
	pts_cfr = zeros(0,2);
	for idx = 1:length(idx_cfr)
		pt = truth(truth(:,1) == idx & truth(:,2) == frame,3:4);
		pts_cfr = [pts_cfr; pt];
	end

	match_gr = [pts_ref, pts_cfr];
	
	%Match ground truth current frame and detections found in current frame
	Gr_fr = pts_cfr;
	t1 = detections(:,4)==frame;
	data1 = detections(t1,:);	
	D_fr = data1(:,2:3);
	opt.viz = 0;
	opt.max_it = 1000;
	[Transform, C_fr] = cpd_register(Gr_fr, D_fr, opt);
	matched = C_fr~=1;
	Gr_fr_m = Gr_fr(C_fr,:);
	match_fr = [Gr_fr_m(matched,:), D_fr(matched,:)];

	%Find correspondences between detections in refframe and detections in current frame

	match_D = matches{frame};
		
	%Then, compute statistics for performance

	%That is, for the manual associations, how many are correct?
	%And, for the detected associations, how many are correct?

	%For each match in detections, convert it to a match in ground truth, see if
	%it is present in the ground truth list

	match_D_gr = zeros(0,4);
	count_D_gr = zeros(size(match_D,1),1);
	for idx = 1:size(match_D, 1)
		m_D_ref = match_D(idx,1:2);
		m_D_fr = match_D(idx,3:4);
		Dgr_ref_row = ismember(floor(match_ref(:,3:4)), floor(m_D_ref), 'rows');
		Dgr_fr_row = ismember(floor(match_fr(:,3:4)), floor(m_D_fr), 'rows');
		if (sum(Dgr_ref_row) == 1) & (sum(Dgr_fr_row) == 1)
			match_D_gr_pt = [match_ref(Dgr_ref_row, 1:2), match_fr(Dgr_fr_row, 1:2)];
		else
			match_D_gr_pt = [-1,-1, -1, -1];
		end
		match_D_gr = [match_D_gr; match_D_gr_pt];
		count_D_gr(idx) = sum(ismember(match_gr, match_D_gr_pt, 'rows'));
	end

	prop_D_gr(frame) = sum(count_D_gr)/size(match_gr,1);
	prop_gr_D(frame) = sum(count_D_gr)/size(match_D,1);

	%For each match in the ground truth, convert it to a match in the detections, 
	%see if it is present in the CPD list
	%match_gr_D = zeros(0,4);
	%count_gr_D = zeros(size(match_gr),1);
	%for idx = 1:size(match_gr, 1)
	%	m_gr_ref = match_gr(idx,1:2);
	%	m_gr_fr = match_gr(idx,3:4);
	%	Grd_ref_row = ismember(floor(match_ref(:,1:2)), floor(m_gr_ref), 'rows');
	%	Grd_fr_row = ismember(floor(match_fr(:,1:2)), floor(m_gr_fr), 'rows');
	%	if (sum(Grd_ref_row) == 1) & (sum(Grd_fr_row) == 1)
	%		match_Gr_d_pt = [match_ref(Grd_ref_row, 3:4), match_fr(Grd_fr_row, 3:4)];
	%	else
	%		match_Gr_d_pt = [-1,-1, -1, -1];
	%	end
	%	match_gr_D = [match_gr_D; match_Gr_d_pt];
	%	count_gr_D(idx) = sum(ismember(floor(match_D), floor(match_Gr_d_pt), 'rows'));
	%end

end

%Save this data for plotting
save(['./tracks/20160412/' name '.mat'], 'matches', 'refframe', 'nF', 'opt', 'prop_gr_D', 'prop_D_gr');

%plot(prop_D_gr); xlabel('frame'); ylabel('proportion of correct detected matches of actual matches')
%figure; plot(prop_gr_D); xlabel('frame'); ylabel('proportion of correct detected matches of predicted matches')
