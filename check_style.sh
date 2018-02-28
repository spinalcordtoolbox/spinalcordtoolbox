#!/usr/bin/env bash

for FILE in `git diff-index --name-only --cached HEAD~1`; do
  if [[ "$FILE" != *.py ]]; then continue; fi;
  echo "Checking $FILE";
  flake8 $FILE;
  if [ $? -ne 0 ]; then exit 255; fi
done;
