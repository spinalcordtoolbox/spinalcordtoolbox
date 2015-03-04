function [energy, jumpPenalty, dataError] = energyL2Potts( u, f, gamma, A, isotropic )
%energyL2Potts Computes the energy of the L2 Potts functional

if exist('A', 'var') && not(isempty(A))
    Au = A * u;
    dataError = sum((Au(:) - f(:)).^2);
else
    dataError = sum((u(:) - f(:)).^2);
end


if isvector(u)
    % vectors
    energy = gamma * diff(u(:)) + dataError;
else
    % matrices
    nJumpComp = 0; % jumps in compass directions
    nJumpDiag = 0; % jumps in diagonal directions
    
    % count jumps
    for i = 1:size(u, 1)
        for j = 1:size(u, 2)-1
            if ~all((u(i,j,:) == u(i,j+1,:)))
                nJumpComp = nJumpComp + 1;
            end
        end
    end
    for i = 1:size(u, 1)-1
        for j = 1:size(u, 2)
            if ~all((u(i,j,:) == u(i+1,j,:)))
                nJumpComp = nJumpComp + 1;
            end
        end
    end
    for i = 1:size(u, 1)-1
        for j = 1:size(u, 2)-1
            if ~all((u(i,j,:) == u(i+1,j+1,:)))
                nJumpDiag = nJumpDiag + 1;
            end
        end
    end
    for i = 1:size(u, 1)-1
        for j = 2:size(u, 2)
            if ~all((u(i,j,:) == u(i+1,j-1,:)))
                nJumpDiag = nJumpDiag + 1;
            end
        end
    end
    
    % set weights (isotropic by default)
    if ~exist('isotropic', 'var') || isotropic
        omega1 = sqrt(2) - 1;
        omega2 = 1 - sqrt(2)/2;
    else
        omega1 = 1;
        omega2 = 0;
    end
    
    % compute energy
    jumpPenalty = (omega1 * nJumpComp + omega2* nJumpDiag);
    energy = gamma * jumpPenalty + dataError;
end