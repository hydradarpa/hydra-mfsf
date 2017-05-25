%  Nonrigid Example 3. Coherent Point Drift (CPD).
%  Nonrigid registration of 3D face point sets with outliers.
%  Full set optioins is explicitelly defined. If you omit some options the
%  default values are used, see help cpd_register.

clear all; close all; clc;
addpath_recurse('~/matlab/CPD2');

%Load data from csv file
fn_in = '../hydra/features/20140829_detections_18_frames.csv';
data = csvread(fn_in, 1, 0);
%ROI,X,Y,T,Contour,Interior,Max. Feret diam.,Ellipse (c),Elongation ratio,Avg. ch 0

%Extract frame 0 and frame 10

t0 = data(:,4)==0;
t10 = data(:,4)==10;

data0 = data(t0,:);
data10 = data(t10,:);

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

%Try first on just X and Y data:
X = data0(:,2:3);
Y = data10(:,2:3);

[Transform, C]=cpd_register(X,Y, opt);


figure,cpd_plot_iter(X, Y); title('Before');
figure,cpd_plot_iter(X, Transform.Y, C);  title('After registering Y to X. And Correspondences');
