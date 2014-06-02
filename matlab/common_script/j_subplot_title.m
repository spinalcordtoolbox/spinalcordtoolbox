function title_handle = subplot_title(title_string)
% subplot_title - displays a title for a subplot, across all subplots
%
% Usage: title_handle = subplot_title(title_string)
%
% Displays a title for a set of subplots at the top of the current figure.
% Said title goes across all of the subplots.  Returns the handle to the
% subplot.
%
% subplot_title is a modified version of code by Keith Rogers and posted
% on comp.soft-sys.matlab on 1995/05/14 in the thread entitled
% "Global title for subplot".
%
% This function is part of froi, available from http://froi.sourceforge.net,
% and is governed under the terms of the Artistic License.
%
% $Id$

ax = gca;
fig = gcf;

title_handle = axes('position',[.1 .90 .8 .05],'Box','off','Visible','off');

title(title_string);
set(get(gca,'Title'),'Visible','On');
set(get(gca,'Title'),'FontSize',10);
set(get(gca,'Title'),'FontWeight','bold');
axes(ax);