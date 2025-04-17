# TODO: Replace "tasks" with a command-line argument passed in by the user
APPTAINER_BIND=' ' apptainer build --build-arg task_installs="tasks" sct.sif sct_model_install.def