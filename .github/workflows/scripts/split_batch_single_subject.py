import argparse
import json
import os
import posixpath
import re
import stat
import sys
from pathlib import Path


HEADER_RE = re.compile(r'^\s*#\s*(?P<title>.+?)\s*$')
SEPARATOR_RE = re.compile(r'^\s*#\s*={5,}\s*$')
DATASET_IN_TITLE_RE = re.compile(r'\(dataset:\s*(?P<dataset>[^\)]+)\)')

# FIXME: List of dataset zip assets published on the sct_tutorial_data
#   GitHub release page.  Any dataset in this set that is NOT referenced by at
#   least one section in batch_single_subject.sh will be reported so it can be
#   manually reviewed and a header added. We probably want to generate this
#   automatically from the release assets using the GitHub API at some point
KNOWN_DATASETS = {
    # "data_batch-processing-of-subjects",
    #     NB1: this one is intentionally excluded: it is a multi-subject
    #          workflow tutorial (using sct_run_batch) that is handled by a separate script/pipeline
    #          and does not belong in batch_single_subject.sh.
    # "data_normalizing-morphometrics-compression",
    #     NB2: this one is intentionally excluded: it is a special case that only
    #          provides pre-generated intermediate files (segmentation, vertebral labels,
    #          compression labels) that users can download to skip the generation steps
    #          already covered by the data_compression section.  All commands that consume
    #          these files are included in the data_compression section, so no separate section
    #          header is needed in batch_single_subject.sh.
    "data_atlas-based-analysis",
    "data_compression",
    "data_contrast-agnostic-registration",
    "data_coregistration",
    "data_gm-wm-metric-computation",
    "data_gm-wm-segmentation",
    "data_improving-registration-with-gm-seg",
    "data_lesion-analysis",
    "data_lumbar-registration",
    "data_ms-lesion-segmentation",
    "data_mtr-computation",
    "data_processing-dmri-data",
    "data_processing-fmri-data",
    "data_rootlets-registration",
    "data_shape-metric-computation",
    "data_spinalcord-segmentation",
    "data_spinalcord-smoothing",
    "data_template-registration",
    "data_vertebral-labeling",
    "data_visualizing-misaligned-cords",
}

# ==============================================================================
#               Split batch_single_subject.sh into mini-scripts
# ==============================================================================
# NB: Sections are defined by the header:
#
#     # Section Title (dataset: data_[DATASET_NAME])
#     # ============================================
#
# Sections with headers that don't contain `(dataset: ...)` are skipped.


def parse_sections(lines):
    sections = []
    skipped = []
    n = len(lines)

    # Find START OF SCRIPT marker present in batch_single_subject.sh to avoid its preamble
    for idx, line in enumerate(lines):
        if 'START OF SCRIPT' in line:
            i = idx + 1
            break
    else:
        raise ValueError('Could not find "START OF SCRIPT" marker in input file. Cannot determine where the actual script starts.')

    while i < n:
        line = lines[i]
        m = HEADER_RE.match(line)
        if m and i + 1 < n and SEPARATOR_RE.match(lines[i + 1]):
            # extract the dataset name and clean it from the title
            title = m.group('title').strip()
            match = DATASET_IN_TITLE_RE.search(title)
            if match:
                dataset = match.group('dataset').strip()
                title = DATASET_IN_TITLE_RE.sub('', title).strip()

            # if we found a header, but it doesn't contain a dataset marker, skip it and all lines until the next header
            else:
                j = i + 2
                while j < n:
                    next_m = HEADER_RE.match(lines[j])
                    if next_m and j + 1 < n and SEPARATOR_RE.match(lines[j + 1]):
                        break
                    j += 1
                skipped.append({'title': title, 'start_line': i + 1})
                print(
                    f'WARNING: skipping section at line {i + 1} '
                    f'"{title}" — no (dataset: ...) marker in header.',
                    file=sys.stderr,
                )
                i = j
                continue

            # collect body until next header+separator or EOF
            body_lines = []
            j = i + 2
            while j < n:
                next_m = HEADER_RE.match(lines[j])
                if next_m and j + 1 < n and SEPARATOR_RE.match(lines[j + 1]):
                    break
                body_lines.append(lines[j])
                j += 1

            sections.append({
                'title': title,
                'dataset': dataset,
                'start_line': i + 1,
                'end_line': j,
                'body': body_lines,
            })
            i = j
        else:
            i += 1

    return sections


# ==============================================================================
#            Handle the `cd` commands that may or may not be present
# ==============================================================================
# Each mini-script is run fresh from `$DATA_DIR`.  The original
# `batch_single_subject.sh` chains all sections together, so a later section
# can write e.g. `cd ../mt` because the previous section left the shell inside
# `t2/`.  That navigation breaks when the section runs in isolation. So, we
# replace every `cd RELATIVE` with `cd "$DATA_DIR/RESOLVED"` so that the
# isolated mini-script always navigates correctly regardless of where the shell
# started.


# NB: regex that matches a live (non-commented) bare `cd TARGET` line.
# - Group 1: leading whitespace
# - Group 2: target path
# - Group 3: trailing whitespace / inline comment (preserved verbatim)
CD_RE = re.compile(r'^(\s*)cd\s+(\S+)([ \t]*(?:#.*)?)$')


def resolve_cd(target, virtual_cwd):
    """Return the new virtual_cwd after `cd TARGET`, rooted at DATA_DIR (='').

    Uses POSIX path arithmetic so that `..` at the root stays at the root
    (same behaviour as a real shell on a dataset root).
    """
    rooted = posixpath.normpath('/' + virtual_cwd + '/' + target)
    rel = rooted.lstrip('/')
    return '' if rel == '.' else rel


def rewrite_cd_commands(body_lines: list, start_cwd: str = '') -> tuple:
    virtual_cwd = start_cwd
    result = []
    for line in body_lines:
        # Leave comment lines alone (they may contain example `cd` commands)
        if line.lstrip().startswith('#'):
            result.append(line)
            continue

        m = CD_RE.match(line)
        if m:
            indent, target, tail = m.group(1), m.group(2), m.group(3)
            # Absolute paths, shell variables ($…), home (~), subshells (`…)
            # — leave them verbatim but still track the cwd change if possible.
            if target[0] in ('/', '$', '~', '`'):
                result.append(line)
                continue

            new_cwd = resolve_cd(target, virtual_cwd)
            virtual_cwd = new_cwd

            if new_cwd:
                result.append(f'{indent}cd "$DATA_DIR/{new_cwd}"{tail}\n')
            else:
                result.append(f'{indent}cd "$DATA_DIR"{tail}\n')
        else:
            result.append(line)
    return result, virtual_cwd


# ==============================================================================
#                   Write the processed mini-scripts to files
# ==============================================================================


# helper functions when generating mini-script files
def make_slug(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", '-', s)
    s = re.sub(r'-+', '-', s).strip('-')
    return s or 'section'


def make_header():
    return """#!/usr/bin/env bash
set -euo pipefail
echo "== Running mini-script generated from batch_single_subject.sh =="
"""


def write_scripts(sections, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    current_cwd = ''
    for idx, section in enumerate(sections, start=1):
        name = section['title']
        dataset = section['dataset']  # always present — sections without dataset are skipped in parse_sections
        slug = make_slug(name)
        dataset_slug = make_slug(dataset)
        fname = f"{idx:02d}_{slug}_{dataset_slug}.sh"
        fpath = outdir / fname
        body = section['body']

        # Determine whether the section body opens with its own `cd` (ignoring
        # blank lines and comments).  If it does not, we must inject the
        # carried cwd so relative file-path arguments in the section commands
        # still resolve correctly.
        section_starts_with_cd = False
        for line in body:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            section_starts_with_cd = bool(CD_RE.match(line))
            break

        with fpath.open('w', newline='\n') as fh:
            fh.write(make_header())
            fh.write('\n')
            fh.write(f'echo "== Section: {name} (dataset: {dataset}) =="\n')
            fh.write('\n')

            # Inject the implied starting directory when the section body has
            # no opening `cd` of its own.  This mirrors the carry-over that
            # the original chained script relies on between sections.
            if not section_starts_with_cd and current_cwd:
                fh.write('# Starting directory carried over from previous section in original script\n')
                fh.write(f'cd "$DATA_DIR/{current_cwd}"\n')
                fh.write('\n')

            # Rewrite relative `cd` commands to use $DATA_DIR-absolute paths,
            # starting from whatever cwd was carried in.
            rewritten, current_cwd = rewrite_cd_commands(body, current_cwd)
            for bl in rewritten:
                fh.write(bl.rstrip('\n') + '\n')

        # FIXME make script executable
        st = os.stat(str(fpath))
        os.chmod(str(fpath), st.st_mode | stat.S_IEXEC)

        manifest.append({
            'name': name,
            'dataset': dataset,
            'file': fname,
            'start_line': section['start_line'],
            'end_line': section['end_line'],
        })

    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', help='Path to batch_single_subject.sh')
    p.add_argument('--out', default='.github/workflows/scripts/batch_single_subject', help='Output directory for mini-scripts')
    args = p.parse_args(argv)

    inp = Path(args.input)
    if not inp.exists():
        print(f'Error: {inp} not found', file=sys.stderr)
        return 2

    lines = inp.read_text(encoding='utf8').splitlines()
    sections = parse_sections(lines)
    if not sections:
        print('No sections with a (dataset: ...) marker detected. Exiting.', file=sys.stderr)
        return 1

    outdir = Path(args.out)
    manifest = write_scripts(sections, outdir)

    manifest_path = outdir / 'sections.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print(f'Wrote {len(manifest)} mini-scripts to {outdir} and manifest to {manifest_path}')

    # Cross-reference datasets found in the script against the known release assets.
    referenced_datasets = {entry['dataset'] for entry in manifest}
    unaccounted = sorted(KNOWN_DATASETS - referenced_datasets)
    if unaccounted:
        print(
            '\nNOTE: The following datasets are published as release assets but have NO '
            'corresponding section (with a dataset: header) in the batch script.\n'
            'These should be reviewed and a new section header added if applicable:',
            file=sys.stderr,
        )
        for ds in unaccounted:
            print(f'  - {ds}', file=sys.stderr)
        print(file=sys.stderr)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
