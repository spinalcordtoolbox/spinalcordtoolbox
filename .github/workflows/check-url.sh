#!/usr/bin/env bash

if [ $# -ne 1 ]; then
    exit 1;
fi

filename=$(cut -d ";" -f 1 <<< "$1")
URL=$(cut -d ";" -f 2 <<< "$1")

if [[ $filename == "./CHANGES.md" ]]; then
  echo "Skipping CHANGES.md ($URL)"
  exit 0
fi

status_code=$(curl --write-out '%{http_code}' --silent  --output /dev/null "$URL")

if [[ $status_code -ge 200 && $status_code -le 299 ]];then
    echo -e "$filename: \x1B[32m✅ OK status code: $status_code for domain $URL  \x1B[0m"
    exit 0
fi

if [[ $status_code -ge 300 && $status_code -le 399 ]];then
    echo -e "$filename: \x1B[33m⚠️ Warning - Redirection - code: $status_code for URL: $URL \x1B[0m"
    exit 1
fi

echo "($status_code) $URL ($filename)" >> invalid_urls.txt
echo -e "$filename: \x1B[31m⛔ Error status code: $status_code for URL: $URL \x1B[0m"
exit 2

