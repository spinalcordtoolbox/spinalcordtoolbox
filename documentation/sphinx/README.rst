#################
SCT Documentation
#################


Scope
#####



Generation
##########

This section considers the following placeholders:

- `${SCT_SOURCE_DIR}`: path to where your SCT extracted archive or git clone (e.g. "./" if you are already located in the git repository)
  resides

- `${SCT_DIR}`: installation path of SCT (eg. `/home/me/sct_3.1.1`)


Prerequisites
*************

There's something important to consider: Sphinx docs are usually generated from
the source tree, and are importing the code.
Meaning you have to have SCT deps installed to run sphinx; this can be
installed by installing SCT and using the SCT python environment to run Sphinx,
from the source directory.

The deps were added as extra docs and aren't installed by default;
to install them:

.. code:: sh

   PATH="${SCT_DIR}/python/bin:$PATH" # we probably want to use SCT's installed python interpreter
   pip install ${SCT_SOURCE_DIR}[docs] # re-install with extras

Sphinx Execution
****************

That's pretty standard:

.. code:: sh

   make --directory=${SCT_SOURCE_DIR}/documentation/sphinx html


