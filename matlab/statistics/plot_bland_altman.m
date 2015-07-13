function [ ] = plot_bland_altman( measures1, measures2 )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% check if 'measures1' and 'measures2' are row vectors (if not transpose it):
if ~isrow(measures1)
    measures1 = measures1';
end
if ~isrow(measures2)
    measures2 = measures2';
end

measures_cat = cat(1,measures2,measures1);

average = mean(measures_cat);

diff_between_measures = diff(measures_cat,1,1);

% STD
mean_diff = mean(diff_between_measures);
std_diff = std(diff_between_measures);

figure
hold on
plot(average, diff_between_measures, 'b.', 'MarkerSize',30.0)
plot(average, mean_diff*ones(1,length(average)),'b-', 'LineWidth',3.0)
plot(average, mean_diff+1.96*std_diff*ones(1,length(average)),'r:')
plot(average, mean_diff-1.96*std_diff*ones(1,length(average)),'r:')
hold off
[xa_nfu,ya_nfu] = ds2nfu([average(1),average(1)],[mean_diff-1.96*std_diff,mean_diff+1.96*std_diff]);
annotation('doublearrow',[0.8,0.8],ya_nfu,'Color','red')
coord_tb = ds2nfu([max(average) mean_diff 1 1]);
annotation('textbox', [0.8 coord_tb(2)+0.01 0.001 0.001],'String', '95%', 'Color','red');
coord_tb2 = ds2nfu([max(average) mean_diff 1 1]);
annotation('textbox', [0.7 ya_nfu(1)-0.001 0.001 0.001],'String', '-1.96.STD', 'Color','red');
annotation('textbox', [0.7 ya_nfu(2)+0.01 0.001 0.001],'String', '+1.96.STD', 'Color','red');
grid minor
xlabel('Average of the 2 measures', 'FontSize', 30.0);
ylabel('Difference between the 2 measures', 'FontSize', 30.0);
title('Bland-Altman plot', 'FontSize', 50.0)
set(gca,'FontSize',20);

end

