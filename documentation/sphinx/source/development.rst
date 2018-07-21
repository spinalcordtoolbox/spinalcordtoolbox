Development Notes
#################


Notes for Core Developers
*************************

Committing and pushing:

- Direct push to the `master` branch is forbidden.


Reporting issues:

- Data that caused the issue: when possible/relevant, copy the data
  there: `duke:sct_testing/issues`

- Please use label: [BUG] & [function] (see list of labels
  [here](https://github.com/neuropoly/spinalcordtoolbox/labels))




Issue Triage
============

- Adjust github labels on the issue report, which might not have
  labels.


Reviewing PRs
=============

The levels of review of a PR are:


#. Basic review with regard to the `PR guidelines
  <https://github.com/neuropoly/spinalcordtoolbox/blob/master/CONTRIBUTING.rst#contributing-code>`_,
  label addition.


#. Review of the code:

   Review the code according to the style, best practices, and the
   “fit” of the code with the rest of SCT.


#. Run-time testing:

   On top of the continuous integration jobs, review that the code
   runs as intended.

   Notes:

   - `batch_processing.sh`, execution on non-public datasets...

   - Only do it after having reviewed the code, even if there is some
     level of accountability, you don't want to run arbitrary code on your system!


#. Scientific review:

   Scientific review of a feature with regard to a paper.


Reviewers should coordinate to cover all of the above.



Binaries Compilation
********************

See https://github.com/neuropoly/spinalcordtoolbox/wiki/Binaries-Compilation

.. TODO


Release Procedure
*****************

.. TODO


Sentry Reporting
****************

`Sentry <https://sentry.io>`_ is used (via the ``raven`` Python
bindings) to report (and prioritize) problems encountered in the
field by SCT users.


Important considerations:


- What is transmitted:

  - Uncaught exceptions

  - Logging messages with a priority of ``ERROR``,
    *except* those who start with `Command-line usage error` as they are
    problems caused by the user and caught by the parser
    (``msct_parser`` or front-end scripts)
    (this is done by monkey-patching the raven logging handler).

- While developing/testing, please disable Sentry reporting to reduce the load
  of those who are watching Sentry reports :)

