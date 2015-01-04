classdef linop
%linop A class for linear operators
    
% written by M. Storath
% $Date: 2014-05-07 11:42:11 +0200 (Mi, 07 Mai 2014) $	$Revision: 89 $

    
    properties
        % function handle for evaluation A * x
        eval;
        % function handle for evaluation A' * x
        ctrans;
        % A'A
        normalOp;
        % true if A'A positive definite
        posdef;
    end
    
    methods
        % constructor
        function A = linop(eval, ctrans, varargin)
            % if it is a matrix
            if ~isa(eval, 'function_handle') 
                A.eval = @(x) eval * x;
                A.ctrans = @(x) eval' * x;
                M = eval' * eval;
                A.normalOp = @(x) M * x;
            else
                A.eval = eval;
                A.ctrans = ctrans;
                A.normalOp = @(x) A.ctrans(A.eval(x));
            end
            ip = inputParser;
            addParamValue(ip, 'posdef', false);
            parse(ip, varargin{:});
            par = ip.Results;
            A.posdef = par.posdef;
        end
        
        % ctranspose (')
        function C = ctranspose(A)
            C =  linop(A.ctrans, A.eval);
        end
        
        % mtimes (*)
        function C = mtimes( A, B )
            if isa(B, 'linop')
                C = linop( @(x) A.eval(B.eval(x)), @(x) B.ctrans(A.ctrans(x)));
            else
                C = A.eval(B);
            end
        end
        
        % size
        function s = size(A)
            %warning('Size is deprecated for class linop');
            s = [1 1];
        end
        
    end
end

