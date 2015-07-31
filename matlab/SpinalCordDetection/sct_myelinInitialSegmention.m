function [initialArray, axonBW] = myelinInitialSegmention(im, axonBW, backBW, verbose,snake,radialProfileMaxLength,khomo_off)
% [initialArray, axonBW] = myelinInitialSegmention(im, axonBW, backBW, verbose,snake)
% exemple: [AxonArray] = myelinInitialSegmention(img, chosenAxon, backBW,1,0);
dbstop if error
if nargin<4, verbose = false; end;
if nargin<5, snake = false; end;

flagAssetsForFigures = 0;%[4:17];
saveAssets = false;
assetsBaseName = '/Users/stevebegin/Documents/PhD/Publication/begin-2014/figures/figure_myelinSeg/1-initialSeg/roi7-120f-med-myelin';

[axonLabel, numAxon] = bwlabel(axonBW);
axonProp = regionprops(axonLabel, 'EquivDiameter');

% numAxon = 10;
initialArray = false(size(im, 1), size(im, 2), numAxon);
throwIdx = false(numAxon, 1);

%% Parameters
numAnglesRadialProfile = 2*36;
radialProfileMinLength = 20; % pixels

if ~exist('radialProfileMaxLength','var'),  radialProfileMaxLength= 2/3; end% fraction of axon EquivDiameter to probe
if ~exist('khomo_off','var'),  khomo_off= 0; end% fraction of axon EquivDiameter to probe

oversamplingFact = 1;

%%
% Meshgrid for computing the radial profiles
[X,Y] = meshgrid(1:size(im, 2),1:size(im, 1));
j_progress('Loop over axons...')
for currentAxonLabel = 1:numAxon
    j_progress(currentAxonLabel/numAxon)
    if verbose && mod(currentAxonLabel, 50) == 0
        fprintf('InitialSegmentation: processing object %i/%i\n', currentAxonLabel, numAxon);
    end
    
    %% Init
    % Get current axon image
    currentAxonBW = (axonLabel == currentAxonLabel);
    if length(find(currentAxonBW))<4
        continue
    end
    % get his radius   
    axonEquivRadius = axonProp(currentAxonLabel).EquivDiameter/2;
    
    % Length of profiles in real image and num pixels in proflies
    radialProfileLength = max([round(axonProp(currentAxonLabel).EquivDiameter*radialProfileMaxLength) radialProfileMinLength]);
    if radialProfileLength == 10;
        oversamplingFact = 3;
    elseif (10 < radialProfileLength) && (radialProfileLength <= 13)
        oversamplingFact = 2;
    else
        oversamplingFact = 1;
    end
    radialProfileNumPix = oversamplingFact*radialProfileLength;
    
    % Arrays
    xCoord = zeros(radialProfileNumPix, numAnglesRadialProfile);
    yCoord = zeros(radialProfileNumPix, numAnglesRadialProfile);
    imRadialProfile = zeros(numAnglesRadialProfile, radialProfileNumPix, 'uint8');
    maskRadialProfile = false(numAnglesRadialProfile, radialProfileNumPix);
    if saveAssets
        axonRadialProfile = false(numAnglesRadialProfile, radialProfileNumPix);
        backRadialProfile = false(numAnglesRadialProfile, radialProfileNumPix);
    end

    %% Compute radial profiles start and end points from axon boundary
    [currentAxonBoundary,~] = bwboundaries(currentAxonBW,'noholes');
    [radialProfileStartpoint, radialProfileEndpoint] = computeProfileEndpoint(fliplr(currentAxonBoundary{1}), radialProfileLength, numAnglesRadialProfile);
    
    %% Compute the radial profiles on the images
    allOtherAxonBW = xor(axonBW, currentAxonBW);
    % Thicken all the axon but the current one to prevent oversegmenting
    allOtherAxonBW = bwmorph(allOtherAxonBW, 'thicken', 1);
    for i=1:numAnglesRadialProfile    
        xCoord(:, i) = linspace(radialProfileStartpoint(i, 1),radialProfileEndpoint(i, 1), radialProfileNumPix)';
        yCoord(:, i) = linspace(radialProfileStartpoint(i, 2),radialProfileEndpoint(i, 2), radialProfileNumPix)';
        imRadialProfile(i, :) = qinterp2(X,Y,double(im),xCoord(:, i)',yCoord(:, i)', 2);
        maskRadialProfile(i, :) = qinterp2(X,Y,double((allOtherAxonBW | backBW)),xCoord(:, i)',yCoord(:, i)', 0);
        if saveAssets
            axonRadialProfile(i, :) = qinterp2(X,Y,double(allOtherAxonBW),xCoord(:, i)',yCoord(:, i)', 0);
            backRadialProfile(i, :) = qinterp2(X,Y,double(backBW),xCoord(:, i)',yCoord(:, i)', 0);
        end
%         maskRadialProfile(i, :) = qinterp2(X,Y,double((allOtherAxonBW)),xCoord(:, i)',yCoord(:, i)', 0);
    end
    if verbose
        anglestoshow=round(linspace(1,numAnglesRadialProfile,5)); anglestoshow=anglestoshow(1:end-1);
        figure(43)
        imPlusbg=imfuse(im,axonBW);
        for i=logspace(3,1.5,10), imzoom(imPlusbg,currentAxonBoundary{1},i), pause(0.1), end
        hold on
        plot([radialProfileStartpoint(anglestoshow,1), radialProfileEndpoint(anglestoshow,1)]', [radialProfileStartpoint(anglestoshow,2), radialProfileEndpoint(anglestoshow,2)]')
        hold off
        figure(44)
        plot(imRadialProfile(anglestoshow,:)')
        title('radial intensity')
    end
    %% Compute horizontal gradient of the profile using Sobel filter
    % detect bordure
    [~,Problematiclength]=find(imRadialProfile==0); Problematiclength=min(min(Problematiclength));
    if (isempty(Problematiclength)) || (Problematiclength==1)
        Problematiclength=radialProfileNumPix;
    end
    % compute gradient
    [imRadialProfileGrad, ~] = imgradientxy(imRadialProfile(:,1:Problematiclength-1), 'Sobel');
    % Image normalisation Scale between 0 and 1
    imRadialProfileGrad = (imRadialProfileGrad - min(imRadialProfileGrad(:)))./(max(imRadialProfileGrad(:)) - min(imRadialProfileGrad(:)));
    if verbose
        figure(45)
        plot(imRadialProfileGrad(anglestoshow,:)')
        title('radial gradients')
    end

%     %% Modify the radial profile for the segmentation
%     % Replace the first value (from left to right, or inside to outside)
%     % coinciding with mask with ONEs
%     imRadialProfileGradMod = imRadialProfileGrad;
%     
%     % Fill the mask from first occurence to end (right)
%     maskRadialProfileFilled = maskRadialProfile;
%     [I, J] = find(maskRadialProfileFilled);
%     [I,ia,~] = unique(I);
%     J = J(ia);
%     for i=1:length(I)
%         maskRadialProfileFilled(I(i), J(i)+1:end) = 1; % Replaced by 1 in imRadialProfileMod
%     end
%     imRadialProfileGradMod(logical(maskRadialProfileFilled)) = 1; % Pushes from the background
%     
%     % If more than 1/3 of a column are ones, replace the whole column by ones.
%     % ind = find((sum(imRadialProfileGradMod == 1)./size(imRadialProfileGradMod, 1)) > 1/3);
%     % if ~isempty(ind)
%     %     imRadialProfileGradMod(:, ind(1):end) = 1;
%     % end
%     
%     % Limit the scope of the segmentation with extreme values of g-ratio
%     minGRatio = 0.2;
%     maxGRatio = 0.85;
%     
%     axonEquivRadius = axonProp(currentAxonLabel).EquivDiameter/2;
%     minProfileLength = ceil(oversamplingFact*(axonEquivRadius/maxGRatio - axonEquivRadius));
%     maxProfileLength = round(oversamplingFact*(axonEquivRadius/minGRatio - axonEquivRadius));    
%     
%     if minProfileLength ~= 0
%         imRadialProfileGradMod(:, 1:minProfileLength) = 1;
%     end
%     if maxProfileLength <= radialProfileLength
%         imRadialProfileGradMod(:, maxProfileLength:end) = 1;
%     end
%     
%     if verbose
%         figure(45)
%         plot(imRadialProfileGradMod(anglestoshow,:)')
%         title('radial gradients')
%     end
    %% Segmentation of the myelin outer profile using 3 methods
%     allPathIdx = zeros(size(imRadialProfileGrad, 1), 3);
%     [sr,allPathIdx(:, 1),~] = minimalPath(imRadialProfileGrad,sqrt(2),verbose);
%     [~,allPathIdx(:, 2:3)] = myelinFindBoundary(imRadialProfileGrad);
% 
%     % Choose the best path based on "straightness" using path standard deviation
%     [~, idxMaxStraight] = min(std(allPathIdx));
%     pathIdx = allPathIdx(:, idxMaxStraight)+1;
% if snake    
% [x1, x2]=as_myelinFindBoundary2(imRadialProfile,axonEquivRadius*2,verbose);
%     x1=round(x1); x2=round(x2);
% else
[~,~,S1]=minimalPath(repmat(imRadialProfileGrad,3,1) ,1, verbose); %ext
S1=S1((numAnglesRadialProfile+1:numAnglesRadialProfile*2),:);
[~,~,S2]=minimalPath(repmat(1-imRadialProfileGrad, 3, 1),1, verbose); %inner
S2=S2((numAnglesRadialProfile+1:numAnglesRadialProfile*2),:);

S1=S1/numAnglesRadialProfile;
S2=S2/numAnglesRadialProfile;

if snake
    
    [x1, x2]=as_myelinFindBoundary2(S2,S1,axonEquivRadius*2,verbose,khomo_off);
    x1=max(0,round(x1));
    x2=min(size(S2,2),round(x2));
else
    S1_reg=minimalpath_reg(S1, axonEquivRadius*(1/(0.99^2)-1) , axonEquivRadius*(1/(0.7^2)-1) , numAnglesRadialProfile);
    
    [~,x1,~] = minimalPath(repmat(S2+S1_reg, 3, 1 ),1,verbose); % find inner bondary
    x1=x1((numAnglesRadialProfile+1:numAnglesRadialProfile*2),:);
    
    S2_reg=zeros(size(S2));
    S2_reg(:,end:-1:1)=minimalpath_reg(S2(:,end:-1:1), axonEquivRadius*(1/(0.99^2)-1) , axonEquivRadius*(1/(0.7^2)-1) , numAnglesRadialProfile);
    
    [~,x2,~] = minimalPath(repmat(S1+S2_reg, 3, 1 ),1, verbose);% find external bondary
    x2=x2((numAnglesRadialProfile+1:numAnglesRadialProfile*2),:);
end


    %% Project the best result back on the original image space

    pathCoordext = zeros(length(x2), 2);
    pathCoordint = zeros(length(x2), 2);
    for i=1:length(x2)
        pathCoordext(i, 1) = xCoord(min(x2(i),size(xCoord,1)), i);
        pathCoordext(i, 2) = yCoord(min(x2(i),size(yCoord,1)), i);
        pathCoordint(i, 1) = xCoord(min(x1(i),size(xCoord,1)), i);
        pathCoordint(i, 2) = yCoord(min(x1(i),size(yCoord,1)), i);
    end
    
    % Smoothing spline 
    % Repeate the profile 3 times to ensure continuity in the spline
    yyext = [[pathCoordext(:, 1)' pathCoordext(:, 1)' pathCoordext(:, 1)']; [pathCoordext(:, 2)' pathCoordext(:, 2)' pathCoordext(:, 2)']];
    yyint = [[pathCoordint(:, 1)' pathCoordint(:, 1)' pathCoordint(:, 1)']; [pathCoordint(:, 2)' pathCoordint(:, 2)' pathCoordint(:, 2)']];
    thext = pi*(0:6/((3*length(pathCoordext)-1)):6);
    thint = pi*(0:6/((3*length(pathCoordint)-1)):6);

    ppext = csaps(thext,yyext, 0.99);
    ppint = csaps(thint,yyint, 0.99);
    yyppext = ppval(ppext, linspace(2*pi,4*pi,100))';    
    yyppint = ppval(ppint, linspace(2*pi,4*pi,100))';    
    %% Transform polygon to binary image
    currentMyelinBW = imfill(poly2mask(yyppext(:, 1), yyppext(:, 2), size(im, 1), size(im, 2)), 'holes');
    currentAxonBW = imfill(poly2mask(yyppint(:, 1), yyppint(:, 2), size(im, 1), size(im, 2)), 'holes');
%     currentMyelinBW = bwmorph(currentMyelinBW, 'majority');
%     currentMyelinBW = bwmorph(currentMyelinBW, 'close');
%     currentMyelinBW = bwmorph(currentMyelinBW, 'thicken', 1);
    currentMyelinBW = xor(currentMyelinBW, currentAxonBW);
    
    currentMyelinBW = bwmorph(currentMyelinBW, 'diag');
    
    %Fill holes except for the axon
    currentMyelinBWFilled = imfill(currentMyelinBW, 'holes');
    currentMyelinBW = xor(currentMyelinBWFilled, currentAxonBW);
    
    initialArray(:, :, currentAxonLabel) = currentMyelinBW;
    axonLabel(axonLabel==currentAxonLabel)=false;
    axonLabel(currentAxonBW)=currentAxonLabel;
    %% Clean Up
    cc = bwconncomp(currentMyelinBW, 4);
    nObjToKeep = 1;
    %            Broken Obj                      Myelin has no holes
    if cc.NumObjects > nObjToKeep || sum(sum(currentAxonBW & currentMyelinBW)) > 0
        throwIdx(currentAxonLabel) = true;
    end
    
    if ~saveAssets && max(flagAssetsForFigures == currentAxonLabel)
        
        linewidth = 2;
        figure;
        set(gcf,'Position',get(0,'ScreenSize'))
        subplot(1, 2, 1)
        pr1 = sc(imRadialProfile) + sc(maskRadialProfile, 'r', maskRadialProfile, 'b', maskRadialProfile)*0.3;
        pr2 = sc(imRadialProfileGrad) + sc((maskRadialProfile==1), 'r', (maskRadialProfile==1), 'b', logical(maskRadialProfile))*0.3;
        pr3 = sc(imRadialProfileGrad) + sc((imRadialProfileGradMod == 1), 'r', (imRadialProfileGradMod == 1))*0.3;
        sc([pr1 pr2 pr3])
        hold on
        plot(x2,sr,'g', 'LineWidth',linewidth);
        plot(x2+radialProfileNumPix,sr,'g', 'LineWidth',linewidth);
        plot(x2+radialProfileNumPix*2,sr,'g', 'LineWidth',linewidth);
        
%         figure;
        subplot(1, 2, 2)
        im1 = sc(im) + sc(axonBW,'b', axonBW, 'g', currentAxonBW, 'r', backBW)*0.5 + sc(currentMyelinBW, 'c', currentMyelinBW)*0.4;
        sc(im1)
%         hold on;
%         plot(yypp(:, 1), yypp(:, 2), 'b', 'LineWidth',linewidth)
        
%         figure;
%         currentMyelinBW2 = imfill(poly2mask(pathCoord(:, 1), pathCoord(:, 2), size(im, 1), size(im, 2)), 'holes');
%         im2 = sc(im) + sc(axonBW,'b', axonBW, 'g', currentAxonBW, 'r', backBW)*0.5 + sc(currentMyelinBW2, 'c', currentMyelinBW2)*0.4;
%         hold on;
%         plot(pathCoord(:, 1), pathCoord(:, 2), 'g', 'LineWidth',linewidth)
%         sc([im1 im2])
    end

    if saveAssets && flagAssetsForFigures == currentAxonLabel
        baseName = [assetsBaseName num2str(currentAxonLabel)];
        %%
        figure;
        sc(sc(im) + sc(currentAxonBW, 'g', currentAxonBW)*0.5);
        if saveAssets, export_fig([baseName '-axonMask.png']); close gcf; end
        %% 
        figure;
        sc(sc(im) + sc(axonBW,'b', axonBW, 'g', currentAxonBW, 'r', backBW)*0.5)
%         sc(sc(im) + sc(axonBW,'b', axonBW, 'g', currentAxonBW)*0.5)
        hold on;
        linewidth = 1.2;
        for i=1:2:numAnglesRadialProfile
            plot([radialProfileStartpoint(i, 1),radialProfileEndpoint(i, 1)], [radialProfileStartpoint(i, 2),radialProfileEndpoint(i, 2)],'Color','k', 'LineWidth',linewidth)
        end
        if saveAssets, export_fig([baseName '-axonMaskRadialLines.pdf']); close gcf; end
        
        %%
        figure;
        % Radial Profile
        sc(sc(imRadialProfile) + sc(maskRadialProfile, 'r', backRadialProfile, 'b', axonRadialProfile)*0.5)
        if saveAssets, export_fig([baseName '-radialProfile.png']); end
        hold on;
        linewidth = 4;
        plot(allPathIdx(:, idxMaxStraight),sr,'g', 'LineWidth',linewidth);
        hold off;
        if saveAssets, export_fig([baseName '-radialProfilePath.png']); close gcf; end
        % Gradient of radial Profile
        figure;
        sc(imRadialProfileGrad)
        if saveAssets, export_fig([baseName '-radialProfileGrad.png']); close gcf; end
        % Gradient of radial Profile with axons and back paint
        figure;
        sc(sc(imRadialProfileGrad) + sc((maskRadialProfile==1), 'r', backRadialProfile, 'b', axonRadialProfile)*0.5)
        if saveAssets, export_fig([baseName '-radialProfileGradMod.png']); end
        hold on
        linewidth = 4;
        plot(allPathIdx(:, idxMaxStraight),sr,'g', 'LineWidth',linewidth);
        if saveAssets, export_fig([baseName '-radialProfileGradModPath.png']); close gcf; end
        % Gradient of radial Profile with axons and back paint median
        figure;
        sc(sc(imRadialProfileGrad) + sc((imRadialProfileGradMod == 1), 'r', (imRadialProfileGradMod == 1))*0.5)
        if saveAssets, export_fig([baseName '-radialProfileGradModMedian.png']); end
        % Same with myelin boundary paths
        hold on
        linewidth = 4;
        plot(allPathIdx(:, idxMaxStraight),sr,'g', 'LineWidth',linewidth);
%         plot(allPathIdx(:, 2),sr,'--b', 'LineWidth',linewidth);
%         plot(allPathIdx(:, 3),sr,'-.r', 'LineWidth',linewidth);
        if saveAssets, export_fig([baseName '-radialProfileMinPath.png']); close gcf; end

        %%
        figure;
        % Unfolded myelin border
        sc(sc(im) + sc(currentMyelinBW, 'c', currentMyelinBW)*0.5);
%         hold on;
%         plot(yypp(:, 1), yypp(:, 2), 'b', 'LineWidth',linewidth)
        if saveAssets, export_fig([baseName '-unfoldSegmentedMyelin.png']); close gcf; end
    end
end

j_progress('elapsed')
%initialArray(:, :, throwIdx) = [];
axonBW = logical(axonLabel);
end

function [startPoint, endPoint] = computeProfileEndpoint(radialProfileStartpoint, radialProfileLength, numAngles)
%% Find the profile endpoints from the boundary normal vectors
% x = radialProfileStartpoint(:, 1);
% y = radialProfileStartpoint(:, 2);

if nargin < 3
    numAngles = 36;
end

% Repeate the profile 3 times to ensure continuity
yy = [[radialProfileStartpoint(:, 1)' radialProfileStartpoint(:, 1)' radialProfileStartpoint(:, 1)']; [radialProfileStartpoint(:, 2)' radialProfileStartpoint(:, 2)' radialProfileStartpoint(:, 2)']];
th = pi*(0:6/((3*length(radialProfileStartpoint)-1)):6);

pp = csaps(th,yy, 0.99);
startPoint = ppval(pp, linspace(2*pi,4*pi,numAngles))'; % eval the middle 1/3 of the spline

x = startPoint(:, 1);
y = startPoint(:, 2);

endPoint = zeros(size(startPoint));
% OLD
% yypp = ppval(pp, th);
% yypp = yypp(:, 1+length(yy)/3:2/3*length(yy)); % grab the middle 1/3 of the spline

% figure;
% plot(yy(1, :), yy(2, :))
% hold on;
% plot(yypp(1, :), yypp(2, :), 'r')

% Gradient on the smoothed spline gives the normal to the boundary
dx = gradient(startPoint(:, 2));
dy = -gradient(startPoint(:, 1));

for i=1:length(dx)
    theta = atan(dx(i)/dy(i));
    %         [dx(i) dy(i) sign(dx(i)) sign(dy(i))]
    
    if (sign(dx(i))*sign(dy(i)) == 1)
        DX = sign(dx(i))*sin(theta)*radialProfileLength;
        DY = sign(dy(i))*cos(theta)*radialProfileLength;
    elseif sign(dx(i))*sign(dy(i)) == 0
        if sign(dy(i)) == 0
            % [x(i) y(i) theta*360/(2*pi)]
            DX = -sin(theta)*radialProfileLength;
            DY = sign(dy(i))*cos(theta)*radialProfileLength;
        else
            DX = -sign(dx(i))*sin(theta)*radialProfileLength;
            DY = sign(dy(i))*cos(theta)*radialProfileLength;
        end
    else
        DY = sign(dy(i))*cos(theta)*radialProfileLength;
        DX = -sign(dx(i))*sin(theta)*radialProfileLength;
    end
    
    endPoint(i, :) = [x(1+mod(i-1, length(x))) + DX y(1+mod(i-1, length(x))) + DY];
    startPoint(i,:) = [x(1+mod(i-1, length(x))) - 0.1*DX y(1+mod(i-1, length(x))) - 0.1*DY];  % update startpoint to be a bit inside.
    if max(max(isnan(endPoint))) | max(max(isnan(startPoint)))
        error('no bound')
    end
    
end

% figure;
% plot(startPoint(:, 1), startPoint(:, 2))
% hold on;
% plot(endPoint(:, 1), endPoint(:, 2), 'r')

end

function [pathIdx, minPath] = myelinFindBoundary(imRadialProfile)

% Prevents from snaping to minima at pixel one
imRadialProfile(:, 1) = 1;

% Init
minPath = ones(size(imRadialProfile, 1), 2);
pathIdx = ones(size(imRadialProfile, 1), 1);

thresh = mean(imRadialProfile(imRadialProfile<1));% - std(imRadialProfile(imRadialProfile<1));
% diffThresh = -0.5;
for i=1:size(imRadialProfile, 1)
    x = imRadialProfile(i, :);
%     diffX = [0 diff(x)];
    minimaIdx = findminima(x);

    minimaUnderThreshIdx = minimaIdx(x(minimaIdx) < thresh);
%     minimaUnderDiffThreshIdx = minimaIdx(diffX(minimaIdx) < diffThresh);
    
    if ~isempty(minimaIdx)
        minPath(i, 1) = minimaIdx(1);
%         if minima(1) > 1 || length(minima) == 1
%             minPath(i, 1) = minima(1);
%         else
%             minPath(i, 1) = minima(2);
%         end
    else
        minPath(i, 1) = nan;
    end
    
    if ~isempty(minimaUnderThreshIdx)
        minPath(i, 2) = minimaUnderThreshIdx(1);
    else
        minPath(i, 2) = nan;
    end

%     if ~isempty(minimaUnderDiffThreshIdx)
%         minPath(i, 3) = minimaUnderDiffThreshIdx(1);
%     else
%         minPath(i, 3) = nan;
%     end

    pathIdx(i, 1) = i;
end

for i=1:size(minPath, 2)
    if find(isnan(minPath(:, i)))
        minPath(isnan(minPath(:, i)), i) = round(nanmedian(minPath(:, i)));
    end
end

% sc(imRadialProfile)
% hold on
% plot(minPath(:, 1), pathIdx, 'b', 'LineWidth',3)
% plot(minPath(:, 2), pathIdx, '--g', 'LineWidth',3)
% plot(minPath(:, 3), pathIdx, '-.r', 'LineWidth',3)

return;
end

