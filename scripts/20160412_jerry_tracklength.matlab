stats_10_50 = tracking_performance_munkres('./tracks/20160412/20160412_dupreannotation_stk0001.csv', './tracks/20160412/jerry_10_50_287.csv', './scripts/20160412_jerry_tracklength_10_50.mat')
stats_15_50 = tracking_performance_munkres('./tracks/20160412/20160412_dupreannotation_stk0001.csv', './tracks/20160412/jerry_15_50_295.csv','./scripts/20160412_jerry_tracklength_15_50.mat')
stats_20_50 = tracking_performance_munkres('./tracks/20160412/20160412_dupreannotation_stk0001.csv', './tracks/20160412/jerry_20_50_309.csv','./scripts/20160412_jerry_tracklength_20_50.mat')
stats_10_70 = tracking_performance_munkres('./tracks/20160412/20160412_dupreannotation_stk0001.csv', './tracks/20160412/jerry_10_70_266.csv','./scripts/20160412_jerry_tracklength_10_70.mat')
stats_15_70 = tracking_performance_munkres('./tracks/20160412/20160412_dupreannotation_stk0001.csv', './tracks/20160412/jerry_15_70_275.csv','./scripts/20160412_jerry_tracklength_15_70.mat')
stats_20_70 = tracking_performance_munkres('./tracks/20160412/20160412_dupreannotation_stk0001.csv', './tracks/20160412/jerry_20_70_288.csv','./scripts/20160412_jerry_tracklength_20_70.mat')

save('./scripts/20160412_jerry_tracklength.mat', 'stats_10_50','stats_15_50','stats_20_50','stats_10_70','stats_15_70','stats_20_70');
load('./scripts/20160412_jerry_tracklength.mat');

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

%Make some plots and stuff
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
