from __future__ import annotations
import typing
if typing.TYPE_CHECKING:
    from typing import ItemsView
    from typing import KeysView


class SampleMetadata(object):
    """Metadata class to help update, get and set metadata values.

    Args:
        d (dict): Initial metadata.

    Attributes:
        metadata (dict): Image metadata.
    """

    def __init__(self, d: dict = None) -> None:
        self.metadata = {} or d

    def __setitem__(self, key: any, value: any) -> None:
        self.metadata[key] = value

    def __getitem__(self, key: any) -> any:
        return self.metadata[key]

    def __contains__(self, key: any) -> bool:
        return key in self.metadata

    def items(self) -> ItemsView:
        return self.metadata.items()

    def _update(self, ref: SampleMetadata, list_keys: list) -> None:
        """Update metadata keys with a reference metadata.

        A given list of metadata keys will be changed and given the values of the reference
        metadata.

        Args:
            ref (SampleMetadata): Reference metadata object.
            list_keys (list): List of keys that need to be updated.
        """
        for k in list_keys:
            if (k not in self.metadata.keys() or not bool(self.metadata[k])) and k in ref.metadata.keys():
                self.metadata[k] = ref.metadata[k]

    def keys(self) -> KeysView:
        return self.metadata.keys()
