#!/usr/bin/env bash
if [ -f ../../.git/hooks/pre-commit ];
then
    rm ../../.git/hooks/pre-commit
fi
cp pre-commit.sh ../../.git/hooks/pre-commit.sh
mv ../../.git/hooks/pre-commit.sh ../../.git/hooks/pre-commit
chmod +x ../../.git/hooks/pre-commit
