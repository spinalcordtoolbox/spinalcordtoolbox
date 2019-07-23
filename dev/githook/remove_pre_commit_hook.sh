#!/usr/bin/env sh
if [[ -f $SCT_DIR/.git/hooks/pre-commit ]];
then
    rm $SCT_DIR/.git/hooks/pre-commit
fi