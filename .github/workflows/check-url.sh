#!/usr/bin/env bash

if [ $# -ne 1 ]; then
    exit 1;
fi

filename=$(cut -d ";" -f 1 <<< "$1")
URL=$(cut -d ";" -f 2 <<< "$1")
# --head: Sends a HEAD request. We use this to be good netizens, since we only need the header to check the response code.
# --silent: Hides the curl progress bar, which is unnecessary noise when testing >600 urls.
# --insecure: Skip SSL verification. Since we're only checking the headers, this should be safe to do. (https://curl.se/docs/sslcerts.html)
CURL_ARGS=(--head --silent --insecure)
# Explicitly write out the HTTP code to stdout, while redirecting the response output to /dev/null
HTTP_CODE_ONLY=(--write-out '%{http_code}' --output /dev/null)
# Override default behavior (exponential backoff + 10m limit) since we don't need that many retries
# We still keep a 5m limit, though, because --retry respects the Retry-After field, which may be greater than 30s.
RETRY_ARGS=(--retry 2 --retry-delay 30 --retry-max-time 300 --retry-all-errors)

# Make sure to check both URL *and* redirections (--location) for excluded patterns
full_info=$(curl "${CURL_ARGS[@]}" --location -- "$URL")
LOCATION=$(curl "${CURL_ARGS[@]}" -- "$URL" | perl -n -e '/^[Ll]ocation: (.*)$/ && print "$1\n"')
# `pipeline-hemis` -> private repository, will 404 (expected)
# `.ru` -> Russian domains, which don't play nicely with curl'ing from GitHub's servers
# `ieeexplore.ieee.org` -> oddly returns a "418 - I'm a teapot" error code instead of 403
# `%s` -> placeholder for a URL, which is used in our documentation's `conf.py` file
if [[ "$full_info + $URL" =~ 'pipeline-hemis'|'.ru'|'ieeexplore.ieee.org'|'%s' ]]; then
    echo -e "$filename: \x1B[33m⚠️  Warning - Skipping: $URL --> $LOCATION\x1B[0m"
    exit 0
fi

# Get the status code for the original URL
status_code=$(curl "${CURL_ARGS[@]}" "${HTTP_CODE_ONLY[@]}" "${RETRY_ARGS[@]}" -- "$URL")

# If there is a redirection, then re-run curl with --location, then continue to check success/failure
if [[ $status_code -ge 300 && $status_code -le 399 ]];then
    echo "($status_code) $URL ($filename)" >> redirected_urls.txt
    echo -e "$filename: \x1B[33m⚠️  Warning - Redirection - code: $status_code for URL $URL --> $LOCATION \x1B[0m"
    status_code=$(curl "${CURL_ARGS[@]}" "${HTTP_CODE_ONLY[@]}" "${RETRY_ARGS[@]}" --location -- "$URL")
    URL=$LOCATION
fi

# Check for success
if [[ $status_code -ge 200 && $status_code -le 299 ]];then
    echo "($status_code) $URL ($filename)" >> valid_urls.txt
    echo -e "$filename: \x1B[32m✅ OK status code: $status_code for domain $URL  \x1B[0m"
    exit 0
fi

# Check for "403 Forbidden" error code (https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/403)
# From https://http.dev/403: "The client does not have access to the requested resource."
# - Due to the rise of AI and LLMs, many sites are now blocking automated access to their content.
# - Often there is no way to bypass this, so in the past we hardcoded the domains to an exclusion list.
#   However, manually editing the blacklist is not a sustainable solution, so we now automatically filter
#   out any sites that raise 403 with just a warning instead of an error.
# - Note: Allowing 403 responses covers the following domains:
#         'twitter.com', 'spiedigitallibrary.org', 'sciencedirect.com', 'wiley.com', 'sagepub.com',
#         'ncbi.nlm.nih.gov', 'oxfordjournals.org', 'docker.com', 'liebertpub.com', 'tandfonline.com',
#         'pnas.org', 'neurology.org', 'academic.oup.com', 'science.org', 'pubs.rsna.org', 'direct.mit.edu',
#         'thejns.org', 'ajnr.org', 'bmj.com', and 'biorxiv.org'
if [[ $status_code -eq 403 ]];then
    echo "($status_code) $URL ($filename)" >> valid_urls.txt
    echo -e "$filename: \x1B[33m⚠️  Warning - Forbidden - status code: $status_code for domain $URL  \x1B[0m"
    exit 0
fi

# Check for "406 Not Acceptable" error code (https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/406)
# From https://http.dev/406: "In practice, this error is rarely used because the server supplies a default
#                             representation instead."
# We encountered this error for the 'Radiopaedia.org' domain.
# After some searching, I believe there are two main causes of this error:
#   - Overly strict 'Accept-*' headers. (In our case, the default header for curl, wget, etc. specifies "accept: */*",
#                                        which is about as open as can be. So I don't think this is the culprit.)
#   - User-agent strings (see e.g. https://stackoverflow.com/a/10043347). (Changing this locally still resulted in 406.)
#
# My best guess is that this due to either A) misconfigured CloudFlare B) a way to prevent LLMs from scraping content.
# So, for now, we just filter out this response, as there's a good chance that this is still accessible via browsers.
if [[ $status_code -eq 406 ]];then
    echo "($status_code) $URL ($filename)" >> valid_urls.txt
    echo -e "$filename: \x1B[33m⚠️  Warning - Not Acceptable - status code: $status_code for domain $URL  \x1B[0m"
    exit 0
fi

# Report failure
echo "($status_code) $URL ($filename)" >> invalid_urls.txt
echo -e "$filename: \x1B[31m⛔ Error status code: $status_code for URL: $URL \x1B[0m"
