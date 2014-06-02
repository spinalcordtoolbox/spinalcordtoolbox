% =========================================================================
% FUNCTION
% j_retrievePath.m
%
% This function retrieve Path and File in an array of char
%
% INPUTS
% arrayin           array of char: (i,path+file(i))
%
% OUTPUTS
% path              path name
% (file)            array of file names
%
% DEPENDANCES
%
% COMMENTS
% Julien Cohen-Adad 2006-11-12
% =========================================================================
function varargout = j_retrievePath(arrayin)


% initializations
if (nargin<1), error('Insufficient arguments. Type "help j_retrievePathFile"'); end

% retrieve path
for i=size(arrayin,2):-1:1
    if (arrayin(1,i)==filesep)
        path_out=arrayin(1,1:i);
        break
    end
end

% retrieve file names
file_out=arrayin(:,i+1:end);

if (nargout==0), error('Insufficient output arguments. Type "help j_retrievePathFile"'); end
if (nargout>0), varargout{1}=path_out; end
if (nargout>1), varargout{2}=file_out; end
