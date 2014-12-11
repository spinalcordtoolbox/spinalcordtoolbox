classdef convop < double
%convop A class for circular convolution by FFT
    
% written by M. Storath
% $Date: 2014-05-07 11:42:11 +0200 (Mi, 07 Mai 2014) $	$Revision: 89 $

    
    methods
        % constructor
        function obj = convop(fourier)
            obj = obj@double(fourier);
        end
        
        % ctranspose (')
        function C = ctranspose(A)
            C = convop(conj(A));
        end
        
        % mtimes (*)
        function C = mtimes( A, B )
            if isa(B, 'convop')
                C = convop( A .* B );
            else
                C = ifftn( A .* fftn(B) );
            end
        end
        
    end
end

