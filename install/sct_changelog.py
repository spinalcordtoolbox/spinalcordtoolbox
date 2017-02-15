"""Create a changelog file from all the merged pull requests

Looking into the latest milestone, all the pull requests for
neuropoly/spinalcordtoolbox are grouped by label and saved in `changlog.md` in
markdown format.

How it works: Once the new tag is ready, you can simply run

`./install/sct_changlog.py`

and copy and paste the content of changlog.[tagId].md to CHANGES.md
"""
import argparse
import logging

import datetime
import requests
import sys

apiURL = 'https://api.github.com/repos/neuropoly/spinalcordtoolbox/'


def latestMilestone():
    """Get from Github the details of the latest milestone
    """
    milestoneURL = apiURL + 'milestones'
    response = requests.get(milestoneURL)
    data = response.json()
    logging.info('Open milestones found %d', len(data))
    logging.info('Latest milestone %s %d', data[0]['title'], data[0]['number'])
    return data[0]


def detailedChangeLog(new_tag):
    """Return the Github URL comparing the last tags with the new_tag.
    """
    tagsURL = apiURL + 'tags'
    response = requests.get(tagsURL)
    previous = response.json()[0]
    return ("https://github.com/neuropoly/spinalcordtoolbox/compare/%s...%s" % (previous['name'], new_tag))


def search(milestone, label=''):
    """Return a list of merged pull requests linked to the milestone and label
    """
    searchAPI = 'https://api.github.com/search/issues'
    query = 'milestone:"%s" is:pr repo:neuropoly/spinalcordtoolbox state:closed is:merged' % (milestone)
    if label:
        query += ' label:%s' % (label)
    payload = {'q': query}
    response = requests.get(searchAPI, payload)
    data = response.json()
    logging.info('Pull requests "%s" labeled %s received %d', milestone, label, len(data))
    return data

def cmd_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tag', help='New tag name', action='store', dest='tag')
    return parser.parse_args()

if __name__ == '__main__':
    options = cmd_options()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='SCT changelog -- %(message)s')
    lines = []
    milestone = latestMilestone()

    lines.append('##{} ({})'.format(milestone['title'], datetime.date.today()))
    lines.append('[View detailed changelog](%s)' % detailedChangeLog(options.tag))

    changelog_pr = set()
    for label in ['bug', 'enhancement', 'feature', 'doc', 'testing']:
        pulls = search(milestone['title'], label)
        if pulls.get('items'):
            lines.append('### %{}'.format(label.upper()))
            changelog_pr = changelog_pr.union(set([x['html_url'] for x in pulls.get('items')]))
            for pull in pulls.get('items'):
                msg = " - (%s) %s [View pull request](%s)" % (pull['id'], pull['title'], pull['html_url'])
                lines.append(msg)

    logging.info('Total pull request in changelog: %d', len(changelog_pr))
    all_pr = set([x['html_url'] for x in search(milestone['title'])['items']])
    diff_pr = all_pr - changelog_pr
    for diff in diff_pr:
        logging.warning('Pull request not labelled: %s', diff)

    filename = 'changelog.%d.md' % milestone['number']
    with open(filename, 'w') as changelog:
        changelog.write('\n'.join(lines))
    logging.info('Changelog saved in %s', filename)
