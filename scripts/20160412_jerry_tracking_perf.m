stats_baseline = tracking_performance_munkres('./tracks/20160412/20160412_dupreannotation_stk0001.csv', './tracks/20160412/jerry_baseline.csv')
stats_error = tracking_performance_munkres('./tracks/20160412/20160412_dupreannotation_stk0001.csv', './tracks/20160412/jerry_error_corrected.csv')
stats_motion = tracking_performance_munkres('./tracks/20160412/20160412_dupreannotation_stk0001.csv', './tracks/20160412/jerry_motion_corrected.csv')
stats_error_motion = tracking_performance_munkres('./tracks/20160412/20160412_dupreannotation_stk0001.csv', './tracks/20160412/jerry_motion_and_error_corrected.csv')

%Save stats
save('./scripts/20160412_jerry_tracking.mat', 'stats_baseline', 'stats_error', 'stats_motion', 'stats_error_motion');

load('./scripts/20160412_jerry_tracking.mat');
