#!/usr/bin/env bash
if [[ -f $SCT_DIR/.git/hooks/pre-commit ]];
then
    rm $SCT_DIR/.git/hooks/pre-commit
fi
cp $SCT_DIR/dev/githook/pre-commit.sh $SCT_DIR/.git/hooks/pre-commit.sh
mv $SCT_DIR/.git/hooks/pre-commit.sh $SCT_DIR/.git/hooks/pre-commit
chmod +x $SCT_DIR/.git/hooks/pre-commit
