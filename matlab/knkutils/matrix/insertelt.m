function v = insertelt(v,pos,x)

% function v = insertelt(v,pos,x)
%
% <v> is a vector
% <pos> is a positive integer referring to a position within <v>.
%   <pos> can be larger than the length of <v>.
% <x> is a scalar or vector
%
% return <v> but with <x> inserted at position <pos>.
% zeros are automatically added if necessary.
%
% if <pos> is [], we do nothing.
%
% example:
% insertelt([4 10 2 4],3,1)

if isempty(pos)
  return;
end
v(pos+length(x):end+length(x)) = v(pos:end);
v(pos:pos+length(x)-1) = x;




%if iscell(f)
%  f{pos:pos+length(x)-1} = x;
%else
%end
