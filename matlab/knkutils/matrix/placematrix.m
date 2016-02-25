function m1 = placematrix(m1,m2,pos)

% function m1 = placematrix(m1,m2,pos)
%
% <m1> is a 2D matrix, with potentially some extra stuff in the third dimension.
% <m2> is a 2D matrix, with potentially some extra stuff in the third dimension.
%   note that size(m1,3) must equal size(m2,3).
% <pos> (optional) is [R C] with a position.  R and C can be any integers.
%   special case is [] which means to center <m2> with respect to <m1>.
%   if exact centering can't be achieved, we shift down and right.
%   default: [].
%
% place <m2> in <m1> positioned with first element at <pos>.
% if any part of <m2> lies outside of <m1>, it just gets ignored.
%
% example:
% isequal(placematrix([1 2 3; 4 5 6; 7 8 9],[10 10; 10 10],[0 0]),[10 2 3; 4 5 6; 7 8 9])

%% SEE ALSO padarray.m ? 
%% see also assignsubvolume2.m???

% input
if ~exist('pos','var') || isempty(pos)
  pos = [];
end

m1r = size(m1,1);
m1c = size(m1,2);
m2r = size(m2,1);
m2c = size(m2,2);

if isempty(pos)
  pos = [1+ceil((m1r-m2r)/2) 1+ceil((m1c-m2c)/2)];
end

badup = max(0,1-pos(1));               % how many bad pixels to the up
badleft = max(0,1-pos(2));             % how many bad pixels to the left
baddown = max(0,(pos(1)+m2r-1)-m1r);   % how many bad pixels to the bottom
badright = max(0,(pos(2)+m2c-1)-m1c);  % how many bad pixels to the right

m1(pos(1)+badup:pos(1)+m2r-1-baddown,pos(2)+badleft:pos(2)+m2c-1-badright,:) = ...
  m2(1+badup:end-baddown,1+badleft:end-badright,:);
