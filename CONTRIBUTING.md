* [Introduction](#introduction)
* [Fixing a bug](#fixing-a-bug)
* [Adding a new feature](#adding-a-new-feature)
* [Commit changes to your branch](#commit-changes-to-your-branch)
* [Submit a pull request](#submit-a-pull-request)
* [Code Review](#code-review)

# Introduction

You can contribute to Spinal Cord Toolbox by opening a Pull Request. Direct push to the `master` branch is forbidden.

If your are new to git or github, the following articles may help you:

* See [Using Pull Requests](https://help.github.com/articles/using-pull-requests) for more information about Pull Requests.
* See [Fork A Repo](http://help.github.com/forking/) for an introduction to forking a repository.
* See [Creating branches](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/) for an introduction on branching within GitHub.
* See [Refining patches using git](https://github.com/erlang/otp/wiki/Refining-patches-using-git) for an introduction to cleaning up git branches.

# Fixing a bug

* In most cases, pull requests for bug fixes should be based on the `master` branch.
* Write a test case before fixing the bug (so that you will know that the test case catches the bug). For applications without a test suite in the git repository, it would be appreciated if you provide a small code sample in the commit message or email a module that will provoke the failure.
* Indicate issue number in the commit (see commit section below)
* Do not close the issue yourself. The issue will be automatically closed when changes are pushed to master.

## Bug reporting template

For issues that are not straight forward, please use the template to structure the bug report:

    Title: [BUG] Summary of the issue. ex:"[BUG] sct_image command crashes when cropping image."

    Environment: Specify what OS and SCT version you are using.
    Step to reproduce: List all the steps that caused the issue.
    Expected results:
    Actual results:
    Data that caused the issue:

# Adding a new feature

* In most cases, pull requests for new features should be based on the master branch.
* It is important to write a good commit message explaining why the feature is needed. We prefer that the information is in the commit message, so that anyone that want to know two years later why a particular feature can easily find out. It does no harm to provide the same information in the pull request (if the pull request consists of a single commit, the commit message will be added to the pull request automatically).
* With few exceptions, it is mandatory to write a new test case that tests the feature. The test case is needed to ensure that the features does not stop working in the future.
* If you are implementing a new feature, also update the documentation to describe the feature.
* Make sure to cite any papers, algorithms or articles that can help understand the implementation of the feature.

## Feature request template

When proposing a new feature, a discussion will be conducted around the feature. Here a good way to present the new feature in the github issues.

    Title: [FEATURE] Summary of the feature.

    Motivation: Explain why the feature is needed.
    Use Case: Explain how the feature will be used, provide all the necessary steps.
    Expected Outcome: What will the outcome be.
    Citation: Provide references to any theoretical work to help the reader better understand the feature.

# Commit changes to your branch

Here are some tips to help the review go smoothly and quickly.

1. Keep it short. Keep the changes less then 50 lines.
2. Focus on committing 1 logical change at a time.
3. Write a verbose commit message. [Detailed explanation of a good commit message](https://github.com/erlang/otp/wiki/writing-good-commit-messages)
4. Correct any code style suggested by an analyser on your changes. [PyCharm](https://www.jetbrains.com/help/pycharm/2016.1/code-inspection.html) has a code analyser integrated or you can use [pyflakes](https://github.com/PyCQA/pyflakes). The only suggestion we are ignoring is the line length.

## Commit message

### Title

The title should be short (50 chars or less), and should explicitly summarize the changes. If it solves an issue, add at the end: "fixes #ISSUE_NUMBER". The message should be preceded by one of the following flags:

```
BUG:   - a change made to fix a runtime issue (crash, segmentation fault, exception, incorrect result)
REF:   - refactoring (edits that don't impact the execution, renaming files, etc.)
OPT:   - a performance improvement, optimization, enhancement
BIN:   - any change related to binary files (should rarely be used)
NEW:   - new functionality added to the project (e.g., new function)
DOC:   - changes not related to the code (comments, documentation, etc.).
TEST:  - any change related to the testing (e.g., sct_test_propseg, .travis, etc.)
DEV:   - related to development (under /dev folder)
INST:  - a modification in the installation/package creation of SCT
```  

An example commit title might be:
```
BUG: Re-ordering of 4th dimension when apply transformation on 4D scans (fixes #1635)
````

### Description

```
Add more detailed explanatory text, if necessary.  Wrap it to about 72
characters or so.  In some contexts, the first line is treated as the
subject of an email and the rest of the text as the body.  The blank
line separating the summary from the body is critical (unless you omit
the body entirely); tools like rebase can get confused if you run the
two together.

Further paragraphs come after blank lines.

  - Bullet points are okay, too

  - Typically a hyphen or asterisk is used for the bullet, preceded by a
    single space, with blank lines in between, but conventions vary here

Solves #1020
```

# Submit a pull request

### Title
The title should be short (50 chars or less), and should explicitly summarize the purpose of the PR. Please note that the PR title is used to automatically generate the [Changelog](https://github.com/neuropoly/spinalcordtoolbox/blob/master/CHANGES.md) for each new release.

### Labels
To help prioritize the request, add labels that describe the type and impact of the change. A change can have multiple types if it is appropriate but would have only 1 impact label. Such as `bug documentation fix:patch`.
Note that labels are used for organizing the Changelog into BUG, ENHANCEMENT, FEATURE, etc.
See [label definitions](https://github.com/neuropoly/spinalcordtoolbox/wiki/Label-definition) on how to categorize the issue and pull request.

[Here](https://github.com/neuropoly/spinalcordtoolbox/pull/1637) is an example of PR with proper labels and description.

# Code Review 

[What is code review?](https://help.github.com/articles/about-pull-request-reviews/)

Any changes submitted to the master branch will go through code review. For a pull request to be accepted it must have:

* At least 2 members approve the changes.
* TravisCI must pass successfully

Reviewing members are :
* @jcohenadad
* @peristeri
* @zougloub
* @benjamindeleener
* @charleygros
* @fperdigon
* @perone
* @stephaniealley
