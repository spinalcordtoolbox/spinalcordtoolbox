function f = circlewithin(c1,c2)

% function f = circlewithin(c1,c2)
%
% <c1> is [x y radius]. can have multiple rows.
% <c2> is [x y radius]
% 
% return <f> as A x 1 with the fraction of circle <c1> that intersects
% circle <c2>.  the output ranges from 0 to 1 inclusive.
%
% example:
% x1 = 7;
% y1 = 9;
% sd1 = 10;
% x2 = 0;
% y2 = 10;
% sd2 = 15;
% figure; hold on;
% drawellipse(x1,y1,0,sd1,sd1,[],[],'r-');
% drawellipse(x2,y2,0,sd2,sd2,[],[],'g-');
% axis equal;
% title(sprintf('what fraction of red intersects green? %.2f',circlewithin([x1 y1 sd1],[x2 y2 sd2])));
%
% history:
% 2010/12/16 - change from binary (all within or not) to a graded response

% center origin at the center of c2
xadj = c2(1);
c2(1) = c2(1) - xadj;
c1(:,1) = c1(:,1) - xadj;
yadj = c2(2);
c2(2) = c2(2) - yadj;
c1(:,2) = c1(:,2) - yadj;

% calc distance to center of c1
dist = sqrt(c1(:,1).^2 + c1(:,2).^2);

% re-position the center of c1 on the x+ axis
c1(:,1) = dist;
c1(:,2) = 0;

% explicitly check for corner cases
bad1 = c2(3) <= c1(:,1)-c1(:,3);   % the two circles do not intersect.  set to 0.
bad2 = c1(:,1)-c1(:,3) <= -c2(3);  % c1 fully encompasses c2.  set to area of c2 divided by area of c1.
bad3 = c1(:,1)+c1(:,3) <= c2(3);   % c1 is fully within c2.  set to 1.

% calculate x-coordinate of the two intersection points
xinter = (c2(3)^2 - c1(:,3).^2 + c1(:,1).^2) ./ (2*c1(:,1));

% calculate the positive y-coordinate of the intersection points
yinter = real(sqrt(c2(3)^2 - xinter.^2));  % force real just to avoid errors from the corner cases

% calculate area of the sliver that sticks out from c2
c2sliver = 2*atan2(yinter,xinter) / (2*pi) .* (pi*c2(3)^2) - xinter.*yinter;

% calculate area of the sliver that sticks out from c1
c1sliver = 2*atan2(yinter,c1(:,1)-xinter) / (2*pi) .* (pi*c1(:,3).^2) - (c1(:,1)-xinter).*yinter;

% calculate total area
areaint = c2sliver + c1sliver;

% finish
f = areaint ./ (pi*c1(:,3).^2);
f(bad1) = 0;
f(bad2) = (pi*c2(3)^2) ./ (pi*c1(bad2,3).^2);
f(bad3) = 1;



% OLD
% % distance from c2 to c1, then add c1's radius.  is this less than or equal to c2's radius?
% f = sqrt((c1(:,1)-c2(1)).^2 + (c1(:,2)-c2(2)).^2) + c1(:,3) <= c2(:,3);
