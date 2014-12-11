classdef radonop < linop
    %radonop A class for Radon transform
    
    % written by M. Storath
    % $Date: 2014-05-07 11:42:11 +0200 (Mi, 07 Mai 2014) $	$Revision: 89 $
    
    properties
        % solution method for prox
        useFBP;
    end
    
    methods
        % constructor
        function A = radonop(theta, imgSize)
            
            eval = @(x) radon(x, theta);
            ctrans = @(x) iradon(x, theta, 'Linear', 'none', imgSize);
            
            A = A@linop(eval, ctrans);
            A.posdef = true;
            A.useFBP = false;
        end
    end
end

