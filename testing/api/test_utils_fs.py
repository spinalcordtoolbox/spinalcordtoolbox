# pytest unit tests for spinalcordtoolbox.utils.fs

from spinalcordtoolbox.utils import fs


def test_cache_signature():
    """Test that input_params are taken into account in the signature."""
    signatures = [
        fs.cache_signature(input_params={}),
        fs.cache_signature(input_params={"version": "1"}),
        fs.cache_signature(input_params={"version": "2"}),
        fs.cache_signature(input_params={"version": "2", "x": "spline"}),
    ]
    assert len(signatures) == len(set(signatures))
