if ismac
	homedir = '/Users/';
else
	homedir = '/home/';
end

addpath([homedir 'lansdell/projects/hydra-mfsf/mfsf']);
addpath([homedir 'lansdell/projects/hydra-mfsf/scripts']);
addpath([homedir 'lansdell/projects/hydra-mfsf/functions']);
addpath_recurse([homedir 'lansdell/matlab/mfsf']);
