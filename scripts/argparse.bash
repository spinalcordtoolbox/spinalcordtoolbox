#!/bin/bash

# 
# Zenput: this allows us to use Python's argparse module to improve some of our simple bash scripts.
# Source: https://github.com/nhoffman/argparse-bash
# 

# Use python's argparse module in shell scripts
#
# The function `argparse` parses its arguments using
# argparse.ArgumentParser; the parser is defined in the function's
# stdin.
#
# Executing ``argparse.bash`` (as opposed to sourcing it) prints a
# script template.
#
# https://github.com/nhoffman/argparse-bash
# MIT License - Copyright (c) 2015 Noah Hoffman

argparse(){
    argparser=$(mktemp 2>/dev/null || mktemp -t argparser)
    cat > "$argparser" <<EOF
from __future__ import print_function
import sys
import argparse
import os


class MyArgumentParser(argparse.ArgumentParser):
    def print_help(self, file=None):
        """Print help and exit with error"""
        super(MyArgumentParser, self).print_help(file=file)
        sys.exit(1)

parser = MyArgumentParser(prog=os.path.basename("$0"))
EOF

    # stdin to this function should contain the parser definition
    cat >> "$argparser"

    # Callers can specify that we shouldn't error out if the user enters an unrecognized flag, in
    # which case we'll use parse_known_args() instead of parse_args() and define a variable called
    # $UNKNOWN_ARGS for the caller
    if [[ $ARGPARSE_ALLOW_UNKNOWN_ARGS ]]; then
        cat >> "$argparser" <<EOF
args, unknown_args = parser.parse_known_args()
print('UNKNOWN_ARGS="{}";'.format(' '.join(unknown_args)))
EOF
    else
        cat >> "$argparser" <<EOF
args = parser.parse_args()
EOF
    fi

    cat >> "$argparser" <<EOF
for arg in [a for a in dir(args) if not a.startswith('_')]:
    key = arg.upper()
    value = getattr(args, arg, None)

    if isinstance(value, bool) or value is None:
        print('{0}="{1}";'.format(key, 'yes' if value else ''))
    elif isinstance(value, list):
        print('{0}=({1});'.format(key, ' '.join('"{0}"'.format(s) for s in value)))
    else:
        print('{0}="{1}";'.format(key, value))
EOF

    # Define variables corresponding to the options if the args can be
    # parsed without errors; otherwise, print the text of the error
    # message.
    if python "$argparser" "$@" &> /dev/null; then
        eval $(python "$argparser" "$@")
        retval=0
    else
        python "$argparser" "$@"
        retval=1
    fi

    rm "$argparser"
    return $retval
}

if echo $0 | grep -q argparse.bash; then
    cat <<FOO
#!/bin/bash

source \$(dirname \$0)/argparse.bash || exit 1
argparse "\$@" <<EOF || exit 1
parser.add_argument('infile')
parser.add_argument('-o', '--outfile')

EOF

echo "INFILE: \${INFILE}"
echo "OUTFILE: \${OUTFILE}"
FOO
fi
