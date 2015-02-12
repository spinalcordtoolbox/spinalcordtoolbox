function setPLJavaPath(static)

% the path
jpath = fullfile( fileparts(which(mfilename)), '..', 'Java', 'bin');

% dynamic path
javaaddpath(jpath);

% static path
if exist('static', 'var') && static
    % static path (is faster, requires restart to take effect)
    upath = userpath;
    jpathfile = fullfile(upath(1:end-1), 'javaclasspath.txt');
    disp(['Appending java class path to ' jpathfile]);
    pathExists = false;
    % check if path exists
    currentPath = javaclasspath('-static');
    
    for i = 1:numel(currentPath)
        pathExists = strcmp(currentPath{i}, jpath) || pathExists;
    end
    % append path to classpath
    if ~pathExists
        try
            fid = fopen(jpathfile, 'at');
            fprintf(fid, '\n');
            % windows path need double backslash
            jpath = strrep(jpath,'\','\\');
            fprintf(fid, jpath);
            fclose(fid);
        catch
            warning('Failed to add java path to static class path. Using dynamic classpath instead. This requires you to call setPLJavaPath.m each time you use Pottslab.')
        end
    end
end



end

