=====================
 Contributing to SCT
=====================


.. contents:: Table of Contents
   :depth: 3
..


Introduction
############

You can contribute to Spinal Cord Toolbox in several ways:

- Reporting issues you encounter

- Providing new code or other content into the SCT repositories

- Contributing to the wiki or mailing list


Reporting a Bug or Requesting a Feature
#######################################


Issues (bugs, feature requests, or others) can be submitted
`on our project's issue page
<https://github.com/neuropoly/spinalcordtoolbox/issues>`_.

.. contents:: See below for guidelines on the steps for opening a
              github issue:
   :local:


Prior to Submitting a New Issue
*******************************

Consider the following:

- Please take a few seconds to search the issue database in case the
  issue has already been raised.

- When reporting an issue, make sure your installation has not been tempered
  with (and if you can, update to the latest release, maybe the problem was
  fixed).


When Submitting an Issue
************************

Issue Title
===========

Try to have a self-descriptive, meaningful issue title,
summarizing the problem you see.

Examples:

- “*installation failure: problem creating launchers*”
- “*sct_crop_image crashes during image cropping*”
- “*add a special mode for squirrel WM/GM segmentation*”


Issue Body
==========

- Describe the issue.

- Provide steps to reproduce the issue.

  Please try to reproduce your issue using ``sct_testing_data`` or
  ``sct_example_data`` as inputs, and to provide a sequence of commands
  that can reproduce it.

  If this is not possible, try to isolate a minimal input on which the issue
  happens (eg. one file among a dataset), and provide this file publicly,
  or if not possible, privately (coordinate with @jcohenadad).

- Add useful information such as screenshots, etc.

- For copy/pasting terminal output (e.g. showing the syntax + error, installation error) please follow these guidelines for clarity:

  - If there is less than 10 lines of text, embed it directly in your comment in github. Use "~~~" to format as code.

  - If there is 10+ lines, either use an `external website
  <https://pastebin.com/>`_, copy/paste your text and include the URL in your comment, or use `collapsable Github markdown capabilities`
  <https://gist.github.com/ericclemmons/b146fe5da72ca1f706b2ef72a20ac39d#using-details-in-github>`_.

- If you submit a feature request, provide a *usage scenario*, imagining
  how the feature would be used (ideally inputs, a sequence of commands,
  and a desired outcome). Also provide references to any theoretical work to help the reader
  better understand the feature.

Issue Labels
============

- To help assigning reviewers and organizing the Changelog, add labels
  that describe the `category <https://github.com/neuropoly/spinalcordtoolbox/wiki/Label-definition#category>`_
  and type of issue.


Issue Examples
==============

Some good real-life examples:

  - https://github.com/neuropoly/spinalcordtoolbox/issues/444


Contributing to the SCT Repository
##################################


Contributions relating to content of the github repository can be
submitted through github pull requests.

.. contents::
   :local:


Prior to Contributing
*********************


Choosing your Baseline
======================


Pull requests for bug fixes or new features should be based on the
`master` branch.


Naming your Branch
==================

When submitting PRs to SCT, please try to follow our convention and
have your branches named as follows:

- Prefix the branch name with a personal identifier and a forward slash;

- If the branch you're working on is in response to an issue, provide
  the issue number;

- Add some text that make the branch name meaningful.

Examples:

- ``cg/propseg-fixup-div0``
- ``jca/1234-rewrite-sct-in-cobol``


Additional Info on Github
=========================

The following github documentation may be of use:

- See `Using Pull Requests
  <https://help.github.com/articles/using-pull-requests>`_
  for more information about Pull Requests.

- See `Fork A Repo <http://help.github.com/forking/>`_ for an
  introduction to forking a repository.

- See `Creating branches
  <https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/>`_
  for an introduction on branching within GitHub.


Developing and committing
*************************

.. contents::
   :local:

Guidelines for developing
=========================

- Make sure the PR changes are not in conflict with the documentation,
  either documentation files (`/README.md`, `/documentation/`), program help,
  SCT Wiki, or SourceForge wiki.

  If conflict, address them.

- Please add tests, especially with new code:

  As of now, we have integration tests (that run in `sct_testing`),
  and unit tests (in `/unit_testing/`).

  They are straightforward to augment, but we understand it's the
  extra mile; it would still be appreciated if you provide something
  lighter (eg. in the commit messages or in the PR or issue text)
  that demonstrates that an issue was fixed, or a feature is functional.

  Consider that if you add test cases, they will ensure that your
  feature -- which you probably care about -- does not stop working
  in the future.

- Please add documentation, if applicable:

  If you are implementing a new feature, also update the
  documentation to describe the feature, and comment the code
  (things that are not trivially understandable from the code)
  to improve its maintainability.

  Make sure to cite any papers, algorithms or articles that can help
  understand the implementation of the feature.
  If you are implementing an algorithm described in a paper,
  add pointers to the section / steps.


- Please review your changes for styling issues, clarity.
  Correct any code style suggested by an analyser on your changes.
  `PyCharm
  <https://www.jetbrains.com/help/pycharm/2016.1/code-inspection.html>`_
  has a code analyser integrated or you can use `pyflakes
  <https://github.com/PyCQA/pyflakes>`_.

  Do not address your functional changes in the same commits as any
  styling clean-up you may be doing on existing code.

- Ensure that you are the original author of your changes,
  and if that is not the case, ensure that the borrowed/adapted code
  is compatible with the SCT MIT license.


Guidelines on Commits
=====================


Commit Titles
+++++++++++++

- Provide a concise and self-descriptive title (avoid > 80 characters)

- You may “scope” the title using the applicable command name(s),
  folder or other "module" as a prefix.

- If a commit is responsible for fixing an issue, post-fix the
  description with ``(fixes #ISSUE_NUMBER)``.

Examples:

- ``testing: add ability to run tests in parallel (fixes #1539)``
- ``deepseg_sc: add utility functions``
- ``documentation: sphinx: add a section about support``
- ``documentation: sphinx: development: fixup typo``
- ``refactor msct_image into image module and compatibility layer``
- ``Travis: remove jobs running Python 2.7``
- ``setup.py: add optional label for installing documentation tooling deps``
- ``testing: add image unit tests``
- ``testing: add sct_deepseg integration tests``


Commit Sequences
++++++++++++++++

- Update your branch to be baseline on the latest master if new
  developments were merged while you were developing.

- **Please prefer `rebasing` to `merging`**, as explained in `this tutorial
  <https://coderwall.com/p/7aymfa/please-oh-please-use-git-pull-rebase>`_.
  Note that if you do rebases after review have started,
  they will be cancelled, so at this point it may be more
  appropriate to do a pull.

- Clean-up your commit sequence. If your are not familiar
  with git, this good tutorial on the subject may help you:
  https://www.atlassian.com/git/tutorials/rewriting-history

- Focus on committing 1 logical change at a time. See `this article
  <https://github.com/erlang/otp/wiki/writing-good-commit-messages>`_
  on the subject.



Submitting a Pull Request
*************************

.. contents::
   :local:


PR Title
========

The PR title is used to automatically generate the `Changelog
<https://github.com/neuropoly/spinalcordtoolbox/blob/master/CHANGES.md>`_
for each new release, so please follow the following rules:

- Provide a concise and self-descriptive title (see `Issue Title`_).

- Do not include the applicable issue number in the title (do it in the `PR Body`_).

- Do not include the function name (use a `PR Labels`_ instead).


PR Body
=======

- Describe what the PR is about, explain the approach and possible drawbacks.
  Don't hesitate to repeat some of the text from the related issue
  (easier to read than having to click on the link).

- If the PR fixes issue(s), indicate it after your introduction:
  ``Fixes #XXXX, Fixes #YYYY``.
  Note: it is important to respect the syntax above so that the issue(s) will be closed upon merging the PR.

- Review the issue according to our documentation in
  `When Submitting an Issue`_.


PR Labels
=========

You **must** add Labels to PRs, as these are used to automatically generate Changelog:

- **Category:** Choose **one** label that describes the
  `category <https://github.com/neuropoly/spinalcordtoolbox/wiki/Label-definition#category>`_
  (white font over purple background).

- **SCT Function:** Choose one or multiple labels corresponding to the SCT functions that are mainly affected by the PR (black font over
  light purple background).

- **Cross-compatibility:** If your PR breaks cross-compatibility with a previous stable release of SCT, you should add the
  label ``compatibility``.

`Here <https://github.com/neuropoly/spinalcordtoolbox/pull/1637>`_ is an example of PR with proper labels and
description.


Continuous Integration
======================

The PR can't be merged if the Travis build hasn't succeeded. If you are familiar with it, consult the Travis test
results and check for possibility of allowed failures.


Reviewers
=========

- Any changes submitted for inclusion to the master branch will have
  to go through a `review
  <https://help.github.com/articles/about-pull-request-reviews/>`_.

- Only request a review when you deem the PR as “good to go”. If the PR is not ready for review, add "(WIP)" at the
  beginning of the title.

- Github may suggest you to add particular reviewers to your PR.
  If that's the case and you don't know better, add all of these suggestions.
  The reviewers will be notified when you add them.
