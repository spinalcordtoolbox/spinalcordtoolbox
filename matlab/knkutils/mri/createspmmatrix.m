function f = createspmmatrix(dim,vox)

% function f = createspmmatrix(dim,vox)
%
% <dim> is like [64 64 16]
% <vox> is like [3 3 3]
%
% return the default SPM transformation matrix that
% goes from matrix space to SPM's internal space.
%
% example:
% createspmmatrix([64 64 16],[3 3 3])

temp = -vox/2-(dim.*vox)/2;
f = [-vox(1) 0 0 -temp(1);
     0 vox(2) 0 temp(2);
     0 0 vox(3) temp(3);
     0 0      0       1];
