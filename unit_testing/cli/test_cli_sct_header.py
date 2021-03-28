import pytest
import logging

from spinalcordtoolbox.scripts import sct_header

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("output_format", ('sct', 'fslhd', 'nibabel'))
def test_sct_header_display_formats_no_checks(output_format):
    """Run the CLI script without checking results."""
    sct_header.main(argv=['display', 'sct_testing_data/t2/t2.nii.gz', '-format', output_format])
