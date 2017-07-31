stats_10_50 = tracking_performance_munkres('./tracks/20160412/20160412_dupreannotation_stk0001.csv', './tracks/20160412/jerry_10_50_287.csv')
stats_15_50 = tracking_performance_munkres('./tracks/20160412/20160412_dupreannotation_stk0001.csv', './tracks/20160412/jerry_15_50_295.csv')
stats_20_50 = tracking_performance_munkres('./tracks/20160412/20160412_dupreannotation_stk0001.csv', './tracks/20160412/jerry_20_50_309.csv')
stats_10_70 = tracking_performance_munkres('./tracks/20160412/20160412_dupreannotation_stk0001.csv', './tracks/20160412/jerry_10_70_266.csv')
stats_15_70 = tracking_performance_munkres('./tracks/20160412/20160412_dupreannotation_stk0001.csv', './tracks/20160412/jerry_15_70_275.csv')
stats_20_70 = tracking_performance_munkres('./tracks/20160412/20160412_dupreannotation_stk0001.csv', './tracks/20160412/jerry_20_70_288.csv')

save('./scripts/20160412_jerry_tracklength.mat', 'stats_10_50','stats_15_50','stats_20_50','stats_10_70','stats_15_70','stats_20_70');
load('./scripts/20160412_jerry_tracklength.mat');


