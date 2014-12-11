%show A show method for numeric data
function showdbl( input, varargin )

if isreal(input)
    %minVal = min(input(:));
    %maxVal = max(input(:));
    if isvector(input)
        % in case of vector, we do a simple plot
        plot(input, varargin{:});
        %title(['Min:' num2str(minVal) ', Max: ' num2str(maxVal) ]);
    elseif ndims(input) == 2
        % in case of 2D-image, we show the image
        colormap bone; 
        imagesc(input,'CDataMapping','scaled', varargin{:});
        axis equal; axis tight;
        %axis image;
        colorbar('location', 'SouthOutside');
        %title(['Black:' num2str(minVal) ', White: ' num2str(maxVal) ]);
    elseif ndims(input) == 3
        if size(input, 3) == 3
            % in case of color 2D-image, we show the colored image
            imagesc(input);
            axis equal; axis tight;
        elseif islogical(input)
            [x y z] = ind2sub(size(input), find(input));
            plot3(y, x, z, 'b.', 'MarkerSize', 0.5);
            grid on;
            axis equal;
        else
            % in case of true 3D-image, we show the central slices
            s = ceil ( size(input) / 2 );
            slice(input, s(2), s(1), s(3), varargin{:});
            shading flat;
            colormap gray; axis equal;
        end
    else
        error('More than 3 dimensions are not supported.')
    end
else
error('Input must be real.');
end

set(gcf, 'color', 'white');

end

