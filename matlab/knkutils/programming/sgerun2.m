function sgerun2(command,name,wantworkspace,jobindices,priority,flags,memusage)

% function sgerun2(command,name,wantworkspace,jobindices,priority,flags,memusage)
%
% <command> (optional) is a string with the MATLAB code to run.
%   default is [] which means to use an interactive prompt to get the code.
% <name> (optional) is a string with the name of the job, like "job_<name>".
%   default is [] which means to use an interactive prompt to get the name.
%   special case is 0 which means generate a random string.
%   valid names are like ^\w+$.
% <wantworkspace> (optional) is whether to make the current workspace
%   available to the code.  default is [] which means to use an interactive
%   prompt to get the answer.  be careful about saving huge workspaces,
%   as this may be inefficient.
% <jobindices> (optional) is:
%   (1) indices of jobs to do.  indices should be consecutive positive integers.  
%       if supplied, we farm out multiple jobs.  the command 
%       that is run by each job is
%         jobindex = X;
%         <command>
%       where X ranges through the elements of <jobindices>.
%   (2) [] which means to do nothing special.
%   default: [].
% <priority> (optional) is an integer in -1024 (lowest) to 1023 (highest).
%   this controls the priority of the job relative to other jobs that you
%   (the user) may have pending.  default: 0.
% <flags> (optional) is a string with additional flags to pass to qsub.
%   this is useful for setting resource requirements for your job.
%   for example, <flags> could be '-l hostname=azure'.  for a list of
%   possible resources, try "qconf -sc".  default: ''.
% <memusage> (optional) is a positive integer indicating the number of megabytes of 
%   memory that you estimate that your job will use.  default: 12000.
%
% the purpose of this function is to make it easy to deploy MATLAB code on the
% Sun Grid Engine (SGE).
%
% basically, what happens is:
%   first, an .m file is created in ~/sgeoutput/ with the code you wish to run, with
%     some various fprintf statements before and after your code.  the file is named 
%     like "job_<name>.m".  
%   then, we use the qsub utility to create an SGE job named like "job_<name>".
%     the actual command associated with the SGE job is "~/matlabsge.sh job_<name>",
%     which, assuming the ~/matlabsge.sh script is setup correctly, simply runs 
%     the MATLAB command "job_<name>".  outputs and errors from the job are
%     written to the ~/sgeoutput/ directory in files named like "job_<name>.oN"
%     and "job_<name>.eN".  the qsub call that we use is written to a text
%     file "job_<name>.command" --- you can use this text file in case you need
%     to re-run the job.
%   if <wantworkspace>, we save the workspace to the ~/sgeoutput/ directory
%     in a file named like "job_<name>.mat", and ensure that the first thing that
%     happens in the SGE job is the loading of this .mat file.
%
% another flag that we use for qsub is -q <queue> where <queue> is the queue to 
% submit the job to.  the default is 'batch.q'.  if you want to submit to a different
% queue, you must run
%   setpref('kendrick','sgequeue','blah.q');
% where blah.q is the name of the queue that you want.  
%
% in order to use sgerun.m, you must prepare the following:
%   1. make the directory ~/sgeoutput/
%   2. make the script ~/matlabsge.sh (a sample version is included below)
%   3. make sure ~/sgeoutput/ is on your MATLAB path when a job is run.
%      because the sgeoutput directory can get big, it might slow down 
%      your MATLAB session if it's actually on the path all the time.
%      so, instead of actually adding the sgeoutput directory to your path,
%      you could simply change to the sgeoutput directory in your
%      matlab startup.m script so that the SGE jobs will be able to
%      see the appropriate .m files when they are executed.
%   4. ensure that the SGE utility qsub is available on the current machine.
%      this might involve adding something like this to your .cshrc file:
%        if (-f /usr/share/gridengine/default/common/settings.csh) then
%          source /usr/share/gridengine/default/common/settings.csh
%        endif 
%
% note that you should clean out ~/sgeoutput/ periodically by deleting the various
% files that are associated with jobs that are no longer needed.  otherwise, lots 
% of files will pile up in the directory and may cause slowdowns.
%
% sample ~/matlabsge.sh script:
% =======================================
% #!/bin/sh
% 
% # call like /bin/sh matlabsge.sh <command>
% #
% # this will run <command> using MATLAB.                                                       
% # edit this script to use the particular version of MATLAB that you desire.
% 
% CMD=$1
% echo Now issuing MATLAB command.
% echo "$CMD" | /hsgs/software/MATLAB-R2012b/bin/matlab -nosplash -nodesktop -nodisplay -singleCompThread
% =======================================
%
% history:
% 2012/10/24 - inherit from sgerun.m

% internal constants
masterdir = '~/sgeoutput/';
masterscript = '~/matlabsge.sh';
numletters = 3;      % number of letters in randomly generated job name
maxsize = 10000000;  % warn if workspace is bigger than this number of bytes

% deal with command input
if ~exist('command','var') || isempty(command)
  fprintf('What command do you wish to run?\n');
  command = inputmulti;
end

% deal with name input
if ~exist('name','var') || isempty(name)
  name = [];
end
if isempty(name)
  while 1
    name = input('What is the name of this job? [default is to randomly generate a name]\n','s');
    if isempty(name)
      name = 0;
      break;
    end
    if isempty(regexp(name,'^\w+$'))
      fprintf('Invalid name. Try again.\n');
    elseif exist([masterdir 'job_' name '.m'],'file') || exist([masterdir 'job_' name '_1.m'],'file')
      fprintf('This job name already exists. Try again.\n');
    else
      break;
    end
  end
end
if isequal(name,0)
  isbad = 1;
  while isbad
    name = randomword(numletters);
    isbad = exist([masterdir 'job_' name '.m'],'file') || exist([masterdir 'job_' name '_1.m'],'file');
  end
end
if exist([masterdir 'job_' name '.m'],'file') || exist([masterdir 'job_' name '_1.m'],'file')
  error('The job name already exists.');
end
name = ['job_' name];
if length(name) > namelengthmax
  error(sprintf('The job name (%s) is too long.',name));
end

% deal with wantworkspace input
if ~exist('wantworkspace','var') || isempty(wantworkspace)
  wantworkspace = input('Do you want the workspace to be available to your job? [0=no [default], 1=yes]\n');
  if isempty(wantworkspace)
    wantworkspace = 0;
  end
  assert(isequal(wantworkspace,0) | isequal(wantworkspace,1));
end

% deal with other inputs
if ~exist('jobindices','var') || isempty(jobindices)
  jobindices = [];
end
if ~exist('priority','var') || isempty(priority)
  priority = 0;
end
if ~exist('flags','var') || isempty(flags)
  flags = '';
end
if ~exist('memusage','var') || isempty(memusage)
  memusage = 12000;
end
assert(isint(priority) && priority >= -1024 && priority <= 1023);
assert(isint(memusage) && memusage >= 1);

% deal with the workspace
if wantworkspace

  % check size of workspace
  a = evalin('caller','whos');
  workspacesize = sum(cat(1,a.bytes));
  
  % give warning if the workspace is big
  if workspacesize > maxsize
    warning('We are saving to disk a workspace that is larger than 10 MB!');
  end

  % save caller's workspace to the special .mat file
  fprintf('saving workspace to disk...');
  evalin('caller',sprintf('save(''%s'');',[masterdir name '.mat']));
  fprintf('done.\n');

  % prepend a load command
  prefix = sprintf(['load(''' masterdir name '.mat''); ']);

else
  prefix = [];
end

% more prefix and suffix stuff
prefix = [prefix 'fprintf(''The job is %s (%s). The current host is %s. The PID is %d. ' ...
                 'The start time is %s.\n'',getenv(''JOB_NAME''),getenv(''JOB_ID''),gethostname,getpid,datestr(now)); '];
suffix = 'fprintf(''The job is %s (%s). The end time is %s.\n'',getenv(''JOB_NAME''),getenv(''JOB_ID''),datestr(now));';

% construct special strings needed for array jobs
if isempty(jobindices)
  specialstr = '';
  tstr = '';
else
  specialstr = 'jobindex = str2double(char(getenv(''SGE_TASK_ID''))); ';
  assert(isequal(jobindices,jobindices(1):jobindices(end)));
  tstr = sprintf('-t %d-%d ',jobindices(1),jobindices(end));
%   else
%     tstr = ['-t ' sprintf('%d,',jobindices)];
%     tstr = tstr(1:end-1);
%   end
end

% write the full command to an .m file
savetext([masterdir name '.m'],{prefix specialstr command suffix});

% deal with queue name
queuename = getpref('kendrick','sgequeue','batch.q');

% construct the command that submits the job
qsubcmd = sprintf('qsub %s -N %s -o %s -e %s -l h_vmem=%dM -p %d %s -S /bin/sh -q %s %s %s; ', ...
  tstr,name,masterdir,masterdir,memusage,priority,flags,queuename,masterscript,name);
%%% -l virtual_free=%dM

% construct the command that makes the .command file
echocmd = sprintf('echo "%s" > ~/sgeoutput/%s.command; ',qsubcmd,name);

% submit the job and make the .command file!
unix_wrapper([qsubcmd echocmd],1);
fprintf('\n\nall jobs successfully created!\n');
