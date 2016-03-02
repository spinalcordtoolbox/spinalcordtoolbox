function f = constructhrfmodelspline(duration,tr,keypoints,seed)

% function f = constructhrfmodelspline(duration,tr,keypoints,seed)
%
% <duration> is the desired HRF duration in seconds
% <tr> is the TR
% <keypoints> is a sorted vector of keypoints that lie within the range (0,<duration>)
% <seed> is vector of the same length as <keypoints> with the initial seed to use
%
% return a spline-based HRF model of the form {A B C D} suitable for use 
% with fitprf.m.  the model consists of using spline interpolation between
% a set of keypoints in order to determine the HRF.  we use a keypoint at 0 s,
% a keypoint at <duration> s, and keypoints at <keypoints>.  the value of the
% keypoint at 0 s and the value of the keypoint at <duration> s are fixed at 0,
% whereas the values of the remaining keypoints are free parameters.  there 
% are no bounds imposed on the free parameters.  also, note that there is an 
% implicit gain flexibility in the model.  the HRF model is evaluated at
% 0*<tr>, 1*<tr>, 2*<tr>, ..., up until a maximum of <duration> s.
%
% example:
% tr = 1.2;
% keypoints = 4:4:48;
% vals = conv2(rand(1,12),[1 1],'same');
% f = constructhrfmodelspline(50,tr,keypoints,zeros(1,12));
% hrf = feval(f{3},vals);
% figure; hold on;
% scatter([0 keypoints 50],[0 vals 0],'k.');
% plot(0:tr:tr*(length(hrf)-1),hrf,'r-');

np = length(keypoints);
f = {seed [repmat(-Inf,[1 np]); repmat(Inf,[1 np])] ...
     @(pp) spline([0 keypoints duration]/tr,[0 pp 0],0:floor(duration/tr))' 1};
