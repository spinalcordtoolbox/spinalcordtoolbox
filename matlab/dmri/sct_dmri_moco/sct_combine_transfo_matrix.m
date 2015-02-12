function slicewise = sct_combine_transfo_matrix(mat_folders, nt, varargin)
% sct_combine_transfo_matrix(mat_folders, [nt, (nz)], (fname_mat_final, fname_log) )
% mat_folders.names      String
% e.g.: mat_folders.names={'./mat_moco/', './mat_eddy/'};
%       sct_combine_transfo_matrix(mat_folders,[666 4])
dbstop if error


if isempty(varargin), fname_mat_final = 'mat_final/'; else fname_mat_final=varargin{1}; end
if length(varargin)>1, log_spline = varargin{2}; else log_spline = 'log_sct_combine_transfo_matrix'; end

% Check if folders are slicewise or not
for imat = 1:length(mat_folders.names)
    if ~isempty(dir([mat_folders.names{imat} '*T*Z*.txt']))
        mat_folders.slicewise(imat)=1;
    else
        mat_folders.slicewise(imat)=0;
    end
end
j_disp(log_spline,['Final matrix are slicewise ? '])
slicewise=max(mat_folders.slicewise);
if slicewise, j_disp(log_spline,['...Yes!']), else j_disp(log_spline,['...No! Volume based']); end

%==========================================================================
%   FIRST MATRIX
%==========================================================================

% create folder for final transfo matrices
if ~exist(fname_mat_final,'dir'), mkdir(fname_mat_final), end

% copy the first registration matrices to the final folder
j_disp(log_spline,['\nCopy the first registration matrices to the final folder...'])
if slicewise == 1 % swape mat_folders.names to have a slicewise element first
    i=find(mat_folders.slicewise,1,'first');
    tmp_folder_name=mat_folders.names{1};
    mat_folders.name{1}=mat_folders.names{i};
    mat_folders.names{i}=tmp_folder_name;
end
cmd = ['cp -r ',mat_folders.names{1},' ',fname_mat_final];
j_disp(log_spline,['>> ',cmd]); [status result] = unix(cmd);


%==========================================================================
%   COMBINE MATRIX
%==========================================================================
j_progress('Combining matrix...')
if slicewise == 0
    for i_mat = 2:length(mat_folders.names)
        j_disp(log_spline,['.. Folder: ',mat_folders(i_mat).name])
        for iT=1:nt(1)
            j_progress(iT/nt(1))
            % Check if matrix exist
            fname_mat_new = [mat_folders(i_mat).name,'mat.T',num2str(iT),'.txt'];
            if exist(fname_mat_new,'file') % Important : b0 matrix might be absent
                % open new matrix
                M_new = textread(fname_mat_new);
                M_new = M_new(1:4,1:4);
            else
                j_disp(log_spline,[fname_mat_new ' doesn''t exist.. no transfo']);
                M_new = diag([1 1 1 1]);
            end
            % open final matrix
            fname_mat_final_T = [fname_mat_final 'mat.T',num2str(iT),'.txt'];
            if exist(fname_mat_final_T)
                M_final = textread(fname_mat_final_T);
                M_final = M_final(1:4,1:4);
            else M_final = diag([1 1 1 1]);
            end
            
            % open new final matrix
            fid = fopen(fname_mat_final_T,'w');
            % initialize new final matrix
            M_new_final = zeros(4,4);
            M_new_final(4,4) = 1;
            % multiply rotation matrices
            M_new_final(1:3,1:3) = M_new(1:3,1:3) * M_final(1:3,1:3);
            % add translation matrices
            M_new_final(1:3,4) = M_new(1:3,4) + M_final(1:3,4);
            % write new final matrix
            fprintf(fid,'%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n',[M_new_final(1,1:4), M_new_final(2,1:4), M_new_final(3,1:4), M_new_final(4,1:4)]);
            fclose(fid);
            
        end
        
    end
    
    
else
    for i_mat = 2:length(mat_folders.names)
        j_disp(log_spline,['.. Folder: ',mat_folders.names{i_mat}])
        for iT=1:nt(1)
            j_progress(iT/nt(1))
            for iZ=1:nt(2)
                if mat_folders.slicewise(i_mat)==1
                    fname_mat_new = [mat_folders.names{i_mat},'mat.T',num2str(iT),'_Z',num2str(iZ),'.txt'];
                else fname_mat_new = [mat_folders.names{i_mat},'mat.T',num2str(iT),'.txt'];
                end
                if exist(fname_mat_new,'file') % Important : b0 matrix might be absent
                    % open new matrix
                    M_new = textread(fname_mat_new);
                    M_new = M_new(1:4,1:4);
                else
                    j_disp(log_spline,[fname_mat_new ' doesn''t exist.. no transfo']);
                    M_new = diag([1 1 1 1]);
                end
                % open final matrix
                fname_mat_final_T_Z = [fname_mat_final 'mat.T',num2str(iT),'_Z',num2str(iZ),'.txt'];
                if exist(fname_mat_final_T_Z)
                    M_final = textread(fname_mat_final_T_Z);
                    M_final = M_final(1:4,1:4);
                else M_final = diag([1 1 1 1]);
                end
                
                % open new final matrix
                fid = fopen(fname_mat_final_T_Z,'w');
                % initialize new final matrix
                M_new_final = zeros(4,4);
                M_new_final(4,4) = 1;
                % multiply rotation matrices
                M_new_final(1:3,1:3) = M_new(1:3,1:3) * M_final(1:3,1:3);
                % add translation matrices
                M_new_final(1:3,4) = M_new(1:3,4) + M_final(1:3,4);
                % write new final matrix
                fprintf(fid,'%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n',[M_new_final(1,1:4), M_new_final(2,1:4), M_new_final(3,1:4), M_new_final(4,1:4)]);
                fclose(fid);
            end
        end
        
    end
end
j_progress('elapsed...')
end