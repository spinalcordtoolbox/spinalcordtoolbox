"""Create a changelog file from all the merged pull requests

Looking into the latest milestone, all the pull requests for neuropoly/spinalcordtoolbox
are grouped by label and saved in `changlog.md` in markdown format.

"""

import logging
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


def detailedChangeLog():
    """Return the Github URL comparing the last two tags.
    """
    tagsURL = apiURL + 'tags'
    response = requests.get(tagsURL)
    latest, previous = response.json()[:2]
    return ("https://github.com/neuropoly/spinalcordtoolbox/compare/%s...%s" % (previous['name'], latest['name']))


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


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='SCT changelog -- %(message)s')
    lines = []
    milestone = latestMilestone()
    lines.append('# Milestone "%s"' % milestone['title'])
    for label in ['bug', 'enhancement', 'feature', 'doc', 'testing', '']:
        lines.append('## %s' % (label.upper() or 'OTHER'))
        lines.append('[View detailed changelog](%s)' % detailedChangeLog())
        pulls = search(milestone['title'], label)
        for pull in pulls.get('items'):
            msg = " - (%s) %s [View pull request](%s)" % (pull['id'], pull['title'], pull['html_url'])
            lines.append(msg)

    filename = 'changelog.%d.md' % milestone['number']
    with open(filename, 'w') as changelog:
        changelog.write('\n'.join(lines))
    logging.info('Changelog saved in %s', filename)
