stats_10_50 = tracking_performance_munkres('./tracks/20160412/20160412_dupreannotation_stk0001.csv', './tracks/20160412/jerry_10_50_287.csv', './scripts/20160412_jerry_tracklength_10_50.mat')
stats_15_50 = tracking_performance_munkres('./tracks/20160412/20160412_dupreannotation_stk0001.csv', './tracks/20160412/jerry_15_50_295.csv','./scripts/20160412_jerry_tracklength_15_50.mat')
stats_20_50 = tracking_performance_munkres('./tracks/20160412/20160412_dupreannotation_stk0001.csv', './tracks/20160412/jerry_20_50_309.csv','./scripts/20160412_jerry_tracklength_20_50.mat')
stats_10_70 = tracking_performance_munkres('./tracks/20160412/20160412_dupreannotation_stk0001.csv', './tracks/20160412/jerry_10_70_266.csv','./scripts/20160412_jerry_tracklength_10_70.mat')
stats_15_70 = tracking_performance_munkres('./tracks/20160412/20160412_dupreannotation_stk0001.csv', './tracks/20160412/jerry_15_70_275.csv','./scripts/20160412_jerry_tracklength_15_70.mat')
stats_20_70 = tracking_performance_munkres('./tracks/20160412/20160412_dupreannotation_stk0001.csv', './tracks/20160412/jerry_20_70_288.csv','./scripts/20160412_jerry_tracklength_20_70.mat')

save('./scripts/20160412_jerry_tracklength.mat', 'stats_10_50','stats_15_50','stats_20_50','stats_10_70','stats_15_70','stats_20_70');
load('./scripts/20160412_jerry_tracklength.mat');

nRT = 620;

%Some stats:
%RMSE
median(stats_10_50.rms(stats_10_50.matched == 1))
median(stats_15_50.rms(stats_15_50.matched == 1))
median(stats_20_50.rms(stats_20_50.matched == 1))

median(stats_10_70.rms(stats_10_70.matched == 1))
median(stats_15_70.rms(stats_15_70.matched == 1))
median(stats_20_70.rms(stats_20_70.matched == 1))

%Length
median(stats_10_50.lifetimes(stats_10_50.matched == 1, 2))
median(stats_15_50.lifetimes(stats_15_50.matched == 1, 2))
median(stats_20_50.lifetimes(stats_20_50.matched == 1, 2))
median(stats_10_70.lifetimes(stats_10_70.matched == 1, 2))
median(stats_15_70.lifetimes(stats_15_70.matched == 1, 2))
median(stats_20_70.lifetimes(stats_20_70.matched == 1, 2))

%
median(stats_10_50.prop_lessthanthr(stats_10_50.matched == 1))
median(stats_15_50.prop_lessthanthr(stats_15_50.matched == 1))
median(stats_20_50.prop_lessthanthr(stats_20_50.matched == 1))
median(stats_10_70.prop_lessthanthr(stats_10_70.matched == 1))
median(stats_15_70.prop_lessthanthr(stats_15_70.matched == 1))
median(stats_20_70.prop_lessthanthr(stats_20_70.matched == 1))

sum(stats_10_50.prop_lessthanthr(stats_10_50.matched == 1)==1)/620
sum(stats_15_50.prop_lessthanthr(stats_15_50.matched == 1)==1)/620
sum(stats_20_50.prop_lessthanthr(stats_20_50.matched == 1)==1)/620
sum(stats_10_70.prop_lessthanthr(stats_10_70.matched == 1)==1)/620
sum(stats_15_70.prop_lessthanthr(stats_15_70.matched == 1)==1)/620
sum(stats_20_70.prop_lessthanthr(stats_20_70.matched == 1)==1)/620

sum(stats_10_50.prop_lessthanthr(stats_10_50.matched == 1)==1)/287
sum(stats_15_50.prop_lessthanthr(stats_15_50.matched == 1)==1)/287
sum(stats_20_50.prop_lessthanthr(stats_20_50.matched == 1)==1)/287
sum(stats_10_70.prop_lessthanthr(stats_10_70.matched == 1)==1)/287
sum(stats_15_70.prop_lessthanthr(stats_15_70.matched == 1)==1)/287
sum(stats_20_70.prop_lessthanthr(stats_20_70.matched == 1)==1)/287

%Make plots with distance per track

stats = {stats_10_50, stats_15_50, stats_20_50, stats_10_70, stats_15_70, stats_20_70};
names = {'gap_10_minlen_50', 'gap_15_minlen_50', 'gap_20_minlen_50', 'gap_10_minlen_70', 'gap_15_minlen_70', 'gap_20_minlen_70'};
nS = length(stats);

fc = figure
hold on 

fd = figure 
hold on

hot(64);
cmap = colormap();
nC = size(cmap,1);

for j = 1:nS
	s = stats{j};
	thr = s.thr;
	figure
	hold on 
	for idx = 1:nRT
		if s.matched(idx) == 1
			cc = ceil(max(s.residual{idx}/40*nC));
			c = cmap(mod(cc,nC)+1,:);
			plot(s.residual{idx}, 'Color', c);
		end
	end
	saveplot(gcf, ['./scripts/20160412_jerry_tracklength_length_' names{j} '.eps'])

	figure
	hold on 
	for idx = 1:nRT
		if s.matched(idx) == 1
			cc = ceil(max(s.residual{idx}/40*nC));
			c = cmap(mod(cc,nC)+1,:);
			plot(s.times{idx}, s.residual{idx}, 'Color', c);
		end
	end
	saveplot(gcf, ['./scripts/20160412_jerry_tracklength_length_' names{j} '.eps'])
	
	distances_count = zeros(1,200);
	distances_ave = zeros(1,200);
	proplessthanthr = zeros(1,200);
	
	%Compute average distances at each time point, compute proportions less than 6 at each time point
	for idx = 1:nRT
		if s.matched(idx) == 1
			t = s.times{idx};
			lessthanthr = s.residual{idx}'<thr;
			distances_ave(1,t) = (distances_ave(1,t).*distances_count(1,t) + s.residual{idx}')./(distances_count(1,t) + 1);
			distances_count(1,t) = distances_count(t) + 1;
			proplessthanthr(1,t) = (proplessthanthr(1,t).*distances_count(1,t) + lessthanthr)./(distances_count(1,t) + 1);
		end
	end
	figure(fc)
	plot(distances_ave)

	figure(fd)
	plot(proplessthanthr)
end
saveplot(gcf, './scripts/20160412_jerry_tracklength_length_distances_ave_time.eps')
saveplot(gcf, './scripts/20160412_jerry_tracklength_length_proplessthr_time.eps')


%Make some histogram plots for statistcs of each track
figure 
subplot(2,3,1)
plot(stats_10_50.prop_lessthanthr(stats_10_50.matched==1), stats_10_50.lifetimes(stats_10_50.matched==1,2), '.')
title('Min length: 50; gap length: 10')
subplot(2,3,2)
plot(stats_15_50.prop_lessthanthr(stats_15_50.matched==1), stats_15_50.lifetimes(stats_15_50.matched==1,2), '.')
title('Min length: 50; gap length: 15')
subplot(2,3,3)
plot(stats_20_50.prop_lessthanthr(stats_20_50.matched==1), stats_20_50.lifetimes(stats_20_50.matched==1,2), '.')
title('Min length: 50; gap length: 20')
subplot(2,3,4)
plot(stats_10_70.prop_lessthanthr(stats_10_70.matched==1), stats_10_70.lifetimes(stats_10_70.matched==1,2), '.')
title('Min length: 70; gap length: 10')
subplot(2,3,5)
plot(stats_15_70.prop_lessthanthr(stats_15_70.matched==1), stats_15_70.lifetimes(stats_15_70.matched==1,2), '.')
title('Min length: 70; gap length: 15')
subplot(2,3,6)
plot(stats_20_70.prop_lessthanthr(stats_20_70.matched==1), stats_20_70.lifetimes(stats_20_70.matched==1,2), '.')
title('Min length: 70; gap length: 20')
saveplot(gcf, './scripts/20160412_jerry_tracklength_length_proplessthr.eps')

figure 
subplot(2,3,1)
plot(stats_10_50.rms(stats_10_50.matched==1), stats_10_50.lifetimes(stats_10_50.matched==1,2), '.')
title('Min length: 50; gap length: 10')
subplot(2,3,2)
plot(stats_15_50.rms(stats_15_50.matched==1), stats_15_50.lifetimes(stats_15_50.matched==1,2), '.')
title('Min length: 50; gap length: 15')
subplot(2,3,3)
plot(stats_20_50.rms(stats_20_50.matched==1), stats_20_50.lifetimes(stats_20_50.matched==1,2), '.')
title('Min length: 50; gap length: 20')
subplot(2,3,4)
plot(stats_10_70.rms(stats_10_70.matched==1), stats_10_70.lifetimes(stats_10_70.matched==1,2), '.')
title('Min length: 70; gap length: 10')
subplot(2,3,5)
plot(stats_15_70.rms(stats_15_70.matched==1), stats_15_70.lifetimes(stats_15_70.matched==1,2), '.')
title('Min length: 70; gap length: 15')
subplot(2,3,6)
plot(stats_20_70.rms(stats_20_70.matched==1), stats_20_70.lifetimes(stats_20_70.matched==1,2), '.')
title('Min length: 70; gap length: 20')
saveplot(gcf, './scripts/20160412_jerry_tracklength_length_rmse.eps')

%Make some plots and stuff
figure 
subplot(2,3,1)
histogram(stats_10_50.prop_lessthanthr(stats_10_50.matched==1), 'Binwidth', .1)
title('Min length: 50; gap length: 10')
subplot(2,3,2)
histogram(stats_15_50.prop_lessthanthr(stats_15_50.matched==1), 'Binwidth', .1)
title('Min length: 50; gap length: 15')
subplot(2,3,3)
histogram(stats_20_50.prop_lessthanthr(stats_20_50.matched==1), 'Binwidth', .1)
title('Min length: 50; gap length: 20')
subplot(2,3,4)
histogram(stats_10_70.prop_lessthanthr(stats_10_70.matched==1), 'Binwidth', .1)
title('Min length: 70; gap length: 10')
subplot(2,3,5)
histogram(stats_15_70.prop_lessthanthr(stats_15_70.matched==1), 'Binwidth', .1)
title('Min length: 70; gap length: 15')
subplot(2,3,6)
histogram(stats_20_70.prop_lessthanthr(stats_20_70.matched==1), 'Binwidth', .1)
title('Min length: 70; gap length: 20')
saveplot(gcf, './scripts/20160412_jerry_tracklength_proplessthanthr.eps')

figure 
subplot(2,3,1)
histogram(stats_10_50.rms(stats_10_50.matched==1), 'Binwidth', 3, 'Binlimits', [0 60])
xlabel('pixels')
title('Min length: 50; gap length: 10')
subplot(2,3,2)
histogram(stats_15_50.rms(stats_15_50.matched==1), 'Binwidth', 3, 'Binlimits', [0 60])
xlabel('pixels')
title('Min length: 50; gap length: 15')
subplot(2,3,3)
histogram(stats_20_50.rms(stats_20_50.matched==1), 'Binwidth', 3, 'Binlimits', [0 60])
xlabel('pixels')
title('Min length: 50; gap length: 20')
subplot(2,3,4)
histogram(stats_10_70.rms(stats_10_70.matched==1), 'Binwidth', 3, 'Binlimits', [0 60])
xlabel('pixels')
title('Min length: 70; gap length: 10')
subplot(2,3,5)
histogram(stats_15_70.rms(stats_15_70.matched==1), 'Binwidth', 3, 'Binlimits', [0 60])
xlabel('pixels')
title('Min length: 70; gap length: 15')
subplot(2,3,6)
histogram(stats_20_70.rms(stats_20_70.matched==1), 'Binwidth', 3, 'Binlimits', [0 60])
xlabel('pixels')
title('Min length: 70; gap length: 20')
saveplot(gcf, './scripts/20160412_jerry_tracklength_rms.eps')

figure 
subplot(2,3,1)
histogram(stats_10_50.lifetimes(stats_10_50.matched==1,2), 'Binwidth', 3, 'Binlimits', [0 102])
xlim([0 102])
xlabel('frames')
title('Min length: 50; gap length: 10')
subplot(2,3,2)
histogram(stats_15_50.lifetimes(stats_15_50.matched==1,2), 'Binwidth', 3, 'Binlimits', [0 102])
xlim([0 102])
xlabel('frames')
title('Min length: 50; gap length: 15')
subplot(2,3,3)
histogram(stats_20_50.lifetimes(stats_20_50.matched==1,2), 'Binwidth', 3, 'Binlimits', [0 102])
xlim([0 102])
xlabel('frames')
title('Min length: 50; gap length: 20')
subplot(2,3,4)
histogram(stats_10_70.lifetimes(stats_10_70.matched==1,2), 'Binwidth', 3, 'Binlimits', [0 102])
xlim([0 102])
xlabel('frames')
title('Min length: 70; gap length: 10')
subplot(2,3,5)
histogram(stats_15_70.lifetimes(stats_15_70.matched==1,2), 'Binwidth', 3, 'Binlimits', [0 102])
xlim([0 102])
xlabel('frames')
title('Min length: 70; gap length: 15')
subplot(2,3,6)
histogram(stats_20_70.lifetimes(stats_20_70.matched==1,2), 'Binwidth', 3, 'Binlimits', [0 102])
xlim([0 102])
xlabel('frames')
title('Min length: 70; gap length: 20')
saveplot(gcf, './scripts/20160412_jerry_tracklength_lifetimes.eps')



%None of the above changes appear to do much to the mean or min RMSE. 
mean(stats_10_50.rms(stats_10_50.matched==1))
mean(stats_15_50.rms(stats_15_50.matched==1))
mean(stats_20_50.rms(stats_20_50.matched==1))

mean(stats_10_70.rms(stats_10_70.matched==1))
mean(stats_15_70.rms(stats_15_70.matched==1))
mean(stats_20_70.rms(stats_20_70.matched==1))

min(stats_10_50.rms(stats_10_50.matched==1))
min(stats_15_50.rms(stats_15_50.matched==1))
min(stats_20_50.rms(stats_20_50.matched==1))

min(stats_10_70.rms(stats_10_70.matched==1))
min(stats_15_70.rms(stats_15_70.matched==1))
min(stats_20_70.rms(stats_20_70.matched==1))
