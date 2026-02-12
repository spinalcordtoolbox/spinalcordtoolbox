#!/usr/bin/env bash
set -eu -o pipefail

printf '%s\n' "Checking for broken links in $# files:" "$@" '=========='

# Find and check each URL in the files provided as arguments to the script.
# The (($# == 0)) test is to prevent grep from using standard input if no files were given.
# The grep -H option is to output the filename even if a single file was given.
# The grep -Zz options are to make the filename and matching URL terminated by null bytes,
# which is the safest way to delimit arbitrary text.
# The (($? == 1)) test is to prevent the script from exiting if no URLs were found.
# Setting IFS= (the empty string) prevents `read` from trimming whitespace.
# The </dev/null redirection is to prevent check-url.sh from stealing the output of grep.
{ (($# == 0)) || grep -EioHZz \
  --exclude 'sct.def' \
  --exclude 'CHANGES.md' \
  --exclude 'requirements.txt' \
  --exclude 'requirements-freeze.txt' \
  '\b(https?)://[-a-z0-9+&@#/%?=~_|$!:,.;]*[a-z0-9+&@#/%=~_|$]' \
  -- "$@" || (($? == 1)) ; } |
  while IFS= read -rd '' filename && IFS= read -rd '' url; do
    .github/workflows/check-url.sh "$filename" "$url" </dev/null
  done

# Summarize the results
touch valid_urls.txt redirected_urls.txt invalid_urls.txt
NUM_OK=$(wc -l valid_urls.txt | cut -d " " -f 1)
NUM_REDIRECT=$(wc -l redirected_urls.txt | cut -d " " -f 1)
NUM_BAD=$(wc -l invalid_urls.txt | cut -d " " -f 1)
echo -en "========== \033[0;32m$NUM_OK passed\033[0;0m, \033[0;33m$NUM_REDIRECT redirected\033[0;0m, \033[0;31m$NUM_BAD failed\033[0;0m ==========\n"
cat invalid_urls.txt

# Exit with failure if there are any bad URLs
((NUM_BAD == 0))
