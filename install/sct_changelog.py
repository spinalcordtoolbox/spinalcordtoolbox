#!/usr/bin/env python
"""Create a changelog file from all the merged pull requests

Looking into the latest github milestone, print out all the pull requests for
neuropoly/spinalcordtoolbox grouped by label and saved in `changlog.[tagId].md`
in markdown format. The command makes the assumption that the milestone title
is formatted as `Release v[MAJOR].[MINOR].[PATCH]`

How it works: Once the new tag is ready, you can simply run

`./install/sct_changlog.py`

and copy and paste the content of changlog.[tagId].md to CHANGES.md

"""
import logging

import datetime
import requests
import sys


API_URL = 'https://api.github.com/repos/neuropoly/spinalcordtoolbox/'


def latest_milestone():
    """Get from Github the details of the latest milestone
    """
    milestone_url = API_URL + 'milestones'
    response = requests.get(milestone_url)
    data = response.json()
    logging.info('Open milestones found %d', len(data))
    logging.info('Latest milestone %s %d', data[0]['title'], data[0]['number'])
    return data[0]


def detailed_changelog(new_tag):
    """Return the Github URL comparing the last tags with the new_tag.
    """
    tags_url = API_URL + 'tags'
    response = requests.get(tags_url)
    previous = response.json()[0]
    return ("https://github.com/neuropoly/spinalcordtoolbox/compare/%s...%s" % (previous['name'], new_tag))


def search(milestone, label=''):
    """Return a list of merged pull requests linked to the milestone and label
    """
    search_url = 'https://api.github.com/search/issues'
    query = 'milestone:"%s" is:pr repo:neuropoly/spinalcordtoolbox state:closed is:merged' % (milestone)
    if label:
        query += ' label:%s' % (label)
    payload = {'q': query}
    response = requests.get(search_url, payload)
    data = response.json()
    logging.info('Pull requests "%s" labeled %s received %d', milestone, label, len(data))
    return data


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='SCT changelog -- %(message)s')
    milestone = latest_milestone()
    title = milestone['title'].split()[-1]

    lines = [
        '## {} ({})'.format(title, datetime.date.today()),
        '[View detailed changelog](%s)' % detailed_changelog(title)
    ]

    changelog_pr = set()
    for label in ['bug', 'enhancement', 'feature', 'doc', 'testing']:
        pulls = search(milestone['title'], label)
        items = pulls.get('items')
        if items:
            lines.append('**{}**'.format(label.upper()))
            changelog_pr = changelog_pr.union(set([x['html_url'] for x in items]))
            items = [" - (%s) %s [View pull request](%s)" % (x['id'], x['title'], x['html_url']) for x in pulls.get('items') ]
            lines.append(items)

    logging.info('Total pull request in changelog: %d', len(changelog_pr))
    all_pr = set([x['html_url'] for x in search(milestone['title'])['items']])
    diff_pr = all_pr - changelog_pr
    for diff in diff_pr:
        logging.warning('Pull request not labelled: %s', diff)

    filename = 'changelog.%d.md' % milestone['number']
    with open(filename, 'w') as changelog:
        changelog.write('\n'.join(lines))
    logging.info('Changelog saved in %s', filename)
