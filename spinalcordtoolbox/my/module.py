import argparse

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers()

my_command1 = subparsers.add_parser('apply', help='Execute provision script, collect all resources and apply them.')

my_command1.add_argument('path', help='Specify path to provision script. provision.py in current'
                                      'directory by default. Also may include url.', default='provision.py')
my_command1.add_argument('-r', '--rollback', action='store_true', default=False, help='If specified will rollback all'
                                                                                      'resources applied.')
my_command1.add_argument('--tree', action='store_true', default=False, help='Print resource tree')
my_command1.add_argument('--dry', action='store_true', default=False, help='Just print changes list')
my_command1.add_argument('--force', action='store_true', default=False, help='Apply without confirmation')
my_command1.add_argument('default_string', default='I am a default', help='Ensure variables are filled in %(prog)s (default %(default)s)')

my_command2 = subparsers.add_parser('game', help='Decision games')
my_command2.add_argument('move', choices=['rock', 'paper', 'scissors'], help='Choices for argument example')
my_command2.add_argument('--opt', choices=['rock', 'paper', 'scissors'], help='Choices for option example')

optional = my_command2.add_argument_group('Group 1')

optional.add_argument('--addition', choices=['Spock', 'lizard'], help='Extra choices for additional group.')
optional.add_argument('--lorem_ipsum',
                      help='Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod '
                           'tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, '
                           'quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo '
                           'consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse '
                           'cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat '
                           'non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.')
