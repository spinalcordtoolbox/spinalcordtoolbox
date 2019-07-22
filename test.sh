Do you want to add the sct_* scripts to your PATH environment? [y]es/[n]o: + read add_to_path
y
+ [[ ! y =~ ^([Yy](es)?|[Nn]o?)$ ]]
+ echo ''

+ [[ y =~ ^[Yy] ]]
+ edit_bashrc
+ [[ -z '' ]]
+ echo

+ echo ''
++ date '+%Y-%m-%d %H:%M:%S'
+ echo '# SPINALCORDTOOLBOX (installed on 2019-07-21 08:14:01)'
+ echo export 'PATH="/Users/gmartinez/dev/spinalcordtoolbox/bin:${PATH}"'
+ echo 'export SCT_DIR=/Users/gmartinez/dev/spinalcordtoolbox'
+ echo 'export MPLBACKEND=Agg'
+ echo ''
+ echo -e 'Validate installation...'
Validate installation...
+ sct_check_dependencies
Traceback (most recent call last):
  File "/Users/gmartinez/dev/spinalcordtoolbox/bin/sct_check_dependencies", line 11, in <module>
    load_entry_point('spinalcordtoolbox', 'console_scripts', 'sct_check_dependencies')()
  File "/Users/gmartinez/dev/spinalcordtoolbox/spinalcordtoolbox/compat/launcher.py", line 34, in main
    os.execvpe(cmd[0], cmd[0:], env)
  File "/Users/gmartinez/dev/spinalcordtoolbox/python/envs/venv_sct/lib/python3.6/os.py", line 568, in execvpe
    _execvpe(file, args, env)
  File "/Users/gmartinez/dev/spinalcordtoolbox/python/envs/venv_sct/lib/python3.6/os.py", line 604, in _execvpe
    raise last_exc.with_traceback(tb)
  File "/Users/gmartinez/dev/spinalcordtoolbox/python/envs/venv_sct/lib/python3.6/os.py", line 594, in _execvpe
    exec_func(fullname, *argrest)
FileNotFoundError: [Errno 2] No such file or directory: b'/Applications/VMware/mpiexec'
+ echo -e 'Installation validation Failed!'
Installation validation Failed!
+ echo -e 'Please copy the historic of this Terminal (starting with the command install_sct) and paste it in the SCT Help forum (create a new discussion):'
Please copy the historic of this Terminal (starting with the command install_sct) and paste it in the SCT Help forum (create a new discussion):
+ echo -e 'http://forum.spinalcordmri.org/c/sct\n'
http://forum.spinalcordmri.org/c/sct

+ exit 1
+ finish
+ value=1
+ cd /Users/gmartinez/dev/spinalcordtoolbox
+ '[' 1 -eq 0 ']'
+ '[' 1 -eq 99 ']'
+ echo -e 'Installation failed\n'
Installation failed

+ rm -r /var/folders/_t/pxl23c8j27jbnnrmsmhl6qwc0001b8/T/tmp.ir7scPQ6