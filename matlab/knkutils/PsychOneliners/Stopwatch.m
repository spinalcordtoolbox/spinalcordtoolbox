% Stopwatch
%
% Time intervals.
%
% Press down the shift key to start the stopwatch.  Release it to stop.
% Use command-period (i.e. apple and period keys) to quit.
%

% HISTORY
% 11/27/02 dgp Wrote it.
% 10/25/05 awi Cosmetic.

fprintf('Use the shift key. Press down to start; release to stop.\n');
fprintf('Hit command-period (i.e. apple and period keys) to quit.\n');
while KbCheck
end

while 1
	keyIsDown=0;
	while ~keyIsDown
		[keyIsDown,timePress,keyCode] = KbCheck;
	end
% 	snd('Play',0.2*sin((0:100000)/10));
	while keyIsDown
		[keyIsDown,timeRelease,keyCode] = KbCheck;
	end
% 	snd('Quiet');
	t=timeRelease-timePress;
	fprintf('%.3f\ts\n',t);
end	
