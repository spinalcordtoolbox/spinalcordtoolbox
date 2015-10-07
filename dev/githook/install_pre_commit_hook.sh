#!/usr/bin/env sh
ln -s pre-commit.sh ../../.git/hooks/pre-commit
chmod 775 ../../.git/hooks/pre-commit
chmod +x ../../.git/hooks/pre-commit
