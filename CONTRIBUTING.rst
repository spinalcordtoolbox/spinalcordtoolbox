
.. contents:: Table of Contents
   :depth: 2
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


Filling-out the Issue Title
===========================

Try to have a self-descriptive, meaningful issue title,
summarizing the problem you see.

Examples:

- “*installation failure: problem creating launchers*”
- “*sct_crop_image crashes during image cropping*”
- “*add a special mode for squirrel WM/GM segmentation*”


Filling-out the Issue Body
==========================

- Describe the issue (the title is probably not enough to explain
  it)

- Provide steps to reproduce the issue.

  Please try to reproduce your issue using SCT example data as
  inputs, and to provide a sequence of commands that can reproduce it.

  If this is not possible, try to isolate a minimal input on which the issue
  happens (eg. one file among a dataset), and to provide this file publicly,
  or if not possible, privately (coordinate with @jcohenadad).

- Feel free to add additional information such as screenshots, etc.


Filling-out the Feature Request Body
====================================

- Try to justify why the feature is desired, and should be in SCT

- If you have an idea of it, try to provide a *usage scenario*, imagining
  how the feature would be used (ideally inputs, a sequence of commands,
  and a desired outcome).

- If you can, provide references to any theoretical work to help the reader
  better understand the feature.


Filling-out Other Fields
========================

- SCT Core Developers must add **Labels** to issues, external developers
  should try to add them.

  To help assigning reviewers and organizing the Changelog, add labels
  that describe the `category <https://github.com/neuropoly/spinalcordtoolbox/wiki/Label-definition#issue-category>`_
  and type of the change.
  A change can have multiple types if it is appropriate but **it can only have one
  category**.

  `Here <https://github.com/neuropoly/spinalcordtoolbox/pull/1637>`_
  is an example of PR with proper labels and description.



More Examples
=============

Some real-life examples:

- Good:

  - https://github.com/neuropoly/spinalcordtoolbox/issues/444


After Submitting an Issue
*************************

Consider the following:

- Please try to be of assistance to help developers with additional
  information, and to confirm resolution of the issue (you may be
  asked to close the issue to confirm).



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

- If you're working on the upstream SCT repository, prefix the branch
  name with a personal identifier and a forward slash;

- If the branch you're working on is in response to an issue, provide
  the issue number;

- Try to add some additional text that make the branch name meaningful
  during its life cycle.

Rationale:

- A merge commit header contains by default the name of the branch to
  be merged

- When working in the main SCT repo, the personal prefix makes the
  branch list prettier and more meaningful.

- As much as Emojis are used a lot in our github, non-ascii branch
  names are not OK (spelling is too hard).

Examples:

- Best:

  - *sct_propseg-fixup-div0*

    - outside contribution assumed
    - can reasonably see what it's about

  - *jca/1234-rewrite-sct-in-cobol*

    - can see who is doing it without looking at the code
    - can see that there is an issue about it
    - can see what it's about and that it's time to change the
      trajectory of this issue

- OK:

  - *jca/1828*

    - you're busy with the science and don't care what the commit log
      will look like or that nobody else will know what this is about,
      nor you in one week... but it's OK since the branch will be
      deleted anyway after merge

- Bad:

  - *wip-on-something*

    - yeah like we can figure that one out without looking at the code

  - *‎‮זאת‬, cJ/😊‎‮sgub-lla-dexif-‬*

    - please no ;)


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


When Developing
***************

.. contents::
   :local:

The Content Itself
==================

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

  Keep in mind that you are responsible for your contributions!


The Commits Metadata
====================


Guidelines on Commit Titles
+++++++++++++++++++++++++++

- Provide a concise and self-descriptive title (especially avoid > 80 characters)

  Some terminology tips:

  - When adding/moving/removing something, use “add/move/remove ...”

  - When correcting a blatant issue, use “fixup ...”

  - When refactoring code, use “refactor ...”

- You may “scope” the title using the applicable command name(s),
  folder or other "module" as a prefix.

- If a commit is responsible for fixing an issue, post-fix the
  description with ``(fixes #ISSUE_NUMBER)``.

- Examples:

  Good:

  - “*sct_testing: add ability to run tests in parallel (fixes #1539)*”
  - “*deepseg_sc: add utility functions*”
  - “*documentation: sphinx: add a section about support*”
  - “*documentation: sphinx: development: fixup typo*”
  - “*refactor msct_image into image module and compatibility layer*”
  - “*Travis: remove jobs running Python 2.7*”
  - “*setup.py: add optional label for installing documentation tooling deps*”
  - “*testing: add image unit tests*”
  - “*testing: add sct_deepseg integration tests*”

  Bad:

  - “*cleaning*” / “*added todo*”: no scope
  - “*Update README.md*”: only scope

Some development operations involve reading commit titles and poor
ones are not helping. Run ``git log --oneline`` and imagine you're
someone else running through it.


Guidelines on Commit Sequences
++++++++++++++++++++++++++++++

- Update your branch to be baselined on the latest master if new
  developments were merged while you were developing.

  See `this tutorial
  <https://coderwall.com/p/7aymfa/please-oh-please-use-git-pull-rebase>`_
  about avoiding using merges within your PR,
  rather rebasing your changes onto the master branch.

  Note that if you do rebases after review have started,
  they will be cancelled, so at this point it may be more
  appropriate to do a pull.

- Try to clean-up your commit sequence.

  If your are not familiar with git, this good tutorial on the
  subject may help you:
  https://www.atlassian.com/git/tutorials/rewriting-history

If the commit sequence is not “clean”, it may make future
developments more costly.


Additional Guidelines on Commits
++++++++++++++++++++++++++++++++

Whole books could be written about that, here are some tips:

- Commit messages are no substitute for in-code documentation.
  The code should be understandable without commit messages,
  commit messages are about explaining *changes*.

- Focus on committing 1 logical change at a time.

- See `this article
  <https://github.com/erlang/otp/wiki/writing-good-commit-messages>`_
  on the subject.



When Submitting your Pull Request
*********************************

.. contents::
   :local:


Filling-out the PR Title
========================

- Provide a concise and self-descriptive title.

  Some terminology tips:

  - When adding something, “add ...”

  - When correcting a blatant issue, use “fixup ...”

- You may “scope” the title using the applicable command name(s),
  folder or other "module" as a prefix.

  Examples:

  - “*sct_testing: add ability to run tests in parallel*”
  - “*documentation: sphinx: add a section about support*”

- Do not include the applicable issue number(s) in the title.

The PR title is used to automatically generate the `Changelog
<https://github.com/neuropoly/spinalcordtoolbox/blob/master/CHANGES.md>`_
for each new release.


Filling-out the PR Body
=======================

- If the PR relates to open issue(s) don't forget to indicate that you are
  fixing them, referring to their number in the PR introduction
  (eg. “This PR fixes #1234”).

  If the PR fixes several issues, please write it as follows: “*Fixes
  #XXXX, Fixes #YYYY, Fixes #ZZZZ*”.

  That syntax will allow to automatically close all the related
  issues upon merging.

  If the issue was opened by a non-core developer, you may elect to
  not use the “fixes #id” syntax to avoid to close the corresponding
  issue automatically, rather request the reporter to confirm
  resolution then close.


- Explain the benefit of merging the PR.

- Explain the approach and possible drawbacks.

  It does not hurt to duplicate/rephrase text coming from the PR commit messages.

- The PR description is no substitute for the commit descriptions.

  Accessing github should not be necessary to figure out that the
  changes brought by a commit are useful.


Other PR Fields
===============

- Take a second look at the commit titles and sequence under the “commits” tab.

  You might notice issues.

- Take a second look at the code changes under “files changed” tab.

  You might notice issues.


- Continuous Integration tests

  The PR can't be merged if the Travis build hasn't succeeded, so
  that's that.

  If you are familiar with it, consult the Travis test results
  and check for possibility of allowed failures.

- Reviewers

  Any changes submitted for inclusion to the master branch will have
  to go through a `review
  <https://help.github.com/articles/about-pull-request-reviews/>`_.

  Only request a review when you deem the PR as “good to go”.

  Github may suggest you to add particular reviewers to your PR.
  If that's the case and you don't know better, add all of these suggestions.

  The reviewers will be notified when you add them.


After Submitting a PR
*********************

Consider that:

- Your collaboration may be requested as part of the PR review process.

- Keep in mind that as the author of a contribution in an free
  software project, you might be contacted about it in the future.
