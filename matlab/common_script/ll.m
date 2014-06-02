% alias to ls -l

function listing = ll(argin)

if nargin == 0
% 	[s,listing] = unix('export LSCOLORS="gxfxcxdxbxegedabagacad"; export CLICOLOR=1; ls -l');
	[s,listing] = unix('ls -l');
else
	[s,listing] = unix(['ls -l ', argin]);
end
