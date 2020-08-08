Development Notes
#################

This section contains additional notes specific to SCT project's organization.


Programming conventions
***********************

See https://github.com/neuropoly/spinalcordtoolbox/wiki#programming-rules

.. TODO


Testing Procedure
*****************

See https://github.com/neuropoly/spinalcordtoolbox/wiki/testing

.. TODO


Release Procedure
*****************

See https://github.com/neuropoly/spinalcordtoolbox/wiki/versioning

.. TODO


Binaries Compilation
********************

See https://github.com/neuropoly/spinalcordtoolbox/wiki/Binaries-Compilation

See https://github.com/neuropoly/spinalcordtoolbox/wiki/Compile-binaries

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

