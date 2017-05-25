names = {'CPD_area', 'CPD_default'};

prop_D_gr_all = zeros(0, nF);
prop_gr_D_all = zeros(0, nF);

for name = names
	load(['./tracks/20160412/' name{1} '.mat']);
	prop_D_gr_all = [prop_D_gr_all; prop_D_gr'];
	prop_gr_D_all = [prop_gr_D_all; prop_gr_D'];
end

plot(1:nF, prop_D_gr_all); xlabel('frame'); ylabel('proportion of correct detected matches of actual matches')
figure; plot(1:nF, prop_gr_D_all); xlabel('frame'); ylabel('proportion of correct detected matches of predicted matches')
