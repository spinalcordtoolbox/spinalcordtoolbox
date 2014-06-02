function j_progress(varargin)
% function fn_progress(prompt,max[,size])
% function fn_progress(prompt,max,'%' or 'p')
% function fn_progress(prompt)
% function fn_progress(i)
% function fn_progress('end')
% function fn_progress('cont')
% from Thomas Deneux, modified by Julien Cohen-Adad 2006-11-24
%
% exemple: j_progress('Pouf pouf. Please wait...');for i=1:50, j_progress(i/50), pause(.01), end;j_progress('elapsed')
%---
% progress indicator

persistent promptsize
persistent curi
persistent after
persistent size
persistent pflag
persistent max

if ischar(varargin{1}) % INITIALIZATION or special
    prompt = varargin{1};
    
    if ischar(prompt) % special cases
        switch prompt
            case 'end'
                fprintf(repmat('\b',1,promptsize+1+size+length(after)+1))
                return
            case 'cont'
                disp(sprintf(repmat(' ',1,promptsize+1+size+length(after))))
                return
            case 'elapsed'
                disp([sprintf([repmat('\b',1,size+length(after)+1) 'OK (elapsed time ' num2str(toc) ' seconds)'])])

%                 disp([sprintf(['elapsed ' num2str(toc) ' s']) after])

%                 disp(['elapsed ' num2str(toc) ' s'])
                return
            case 'elapsedmin'
                disp(['elapsed ' num2str(toc/60,'%.1f') 's.'])
                return
        end
    end
    
    % Input
    promptsize = length(prompt);
    i = 0; curi = i;
    if nargin<2
        pflag = true;
        max = 1;
        size = 3;
    else
        max = varargin{2};
        if ischar(max)
            max = str2num(max); 
        end
        pflag = false;
        if nargin>2
            size = varargin{3};
            if strcmp(size,'%') || strcmp(size,'p')
                pflag = true;
                size = 3;
            elseif ischar(size)
                size = str2num(size); 
            end
        else
            size = floor(log10(max))+1;
        end
    end
    csize = num2str(size);
    if pflag
        after = sprintf('%%');
    else
        after = sprintf(['/%' csize 'i'],max);
    end
    disp([sprintf([prompt ' %' csize 'i'],i) after])
    tic
else % STATE PROGRESS
    i = varargin{1};
    if pflag, i = floor(i/max*100); end
    if (i == curi), return, end
    curi = i;
    csize = num2str(size);
    disp([sprintf([repmat('\b',1,size+length(after)+1) '%' csize 'i'],i) after])
end