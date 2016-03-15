function printnice(figs,mode,directory,prefix)

% function printnice(figs,mode,directory,prefix)
%
% <figs> (optional) is a vector of figure numbers.  default: [gcf].
% <mode> (optional) is
%   0 means .eps (using flags "-depsc2 -painters -r300").
%     [0 1] means also use the flag "-loose".
%   [1 n] means .png at n pixels per inch (using flags "-dpng -r(n)")
%   default: 0.
% <directory> (optional) is the directory to temporarily change into
%   when writing the files.  default is the current working directory.
%   we automatically make the directory if it doesn't exist.
% <prefix> (optional) is the prefix of the output filename.  the prefix
%   can include in it '%d' (or a variant thereof) for the figure number.
%   you can pass in a number and we automatically convert it using num2str.
%   default: '%d'.
%
% print figure windows to files in <directory>.
%
% note that if <prefix> has a directory that precedes the actual filename,
% we attempt to automatically make that directory.
%
% history:
% 2011/06/29 - temporarily change PaperPositionMode to auto before printing
%
% example:
% figure; scatter(randn(100,1),randn(100,1),'r.'); printnice;

% NOTE: removed special eps pixel mode (see old printnice.m)
% SEE: figurewrite.m

% input
if ~exist('figs','var') || isempty(figs)
  figs = [gcf];
end
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end
if ~exist('directory','var') || isempty(directory)
  directory = pwd;
end
if ~exist('prefix','var') || isempty(prefix)
  prefix = '%d';
end
if ~ischar(prefix)
  prefix = num2str(prefix);
end

  olddir = pwd;
  mkdirquiet(directory);
  cd(directory);

% make dir if necessary
dir0 = stripfile(prefix);
if ~isempty(dir0) && ~exist(dir0,'dir')
  mkdirquiet(dir0);
end

% do it
for p=1:length(figs)
  fig = figs(p);

  % temporarily change
  prev = get(fig,'PaperPositionMode');
  set(fig,'PaperPositionMode','auto');

  switch mode(1)
  case 0
    filename = sprintf([prefix '.eps'],fig);
    if length(mode) > 1
      print(fig,'-depsc2','-painters','-r300','-loose',filename);
    else
      print(fig,'-depsc2','-painters','-r300',filename);
    end
  case 1
    filename = sprintf([prefix '.png'],fig);
    print(fig,'-dpng',['-r' num2str(mode(2))],filename);  % painters, zbuffer, opengl???  what is correct?
  end
%  fprintf('wrote %s.\n',filename);

  % restore
  set(fig,'PaperPositionMode',prev);

end

  cd(olddir);
