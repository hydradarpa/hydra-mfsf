if ismac
	homedir = '/Users/';
else
	homedir = '/home/';
end

addpath([homedir 'lansdell/projects/hydra-mfsf/mfsf']);
addpath_recurse([homedir 'lansdell/matlab/mfsf']);
