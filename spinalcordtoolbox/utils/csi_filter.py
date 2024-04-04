"""
Color code removal filter, implemented as a codec

Should be usable for writing to a file, for example:

>>> from spinalcordtoolbox.utils import csi_filter
>>> csi_filter.register_codec()
>>> with open('some_file', 'w', encoding='csi-filter') as some_file:
...     some_file.write('Some \x1b[1mBRIGHT\x1b[m text\n')
24
>>> open('some_file', 'rb').read()
'Some BRIGHT text\n'

Copyright (c) 2024 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import codecs
import re

# a complete CSI escape sequence (cursor movements, color codes, etc.)
csi_escape = re.compile(
    r"""
    \x1b \[  # the control sequence introducer, ESC [
    [0-?]*   # optional parameter bytes, between 0x30 and 0x3f
    [ -/]*   # optional intermediate bytes, between 0x20 and 0x2f
    [@-~]    # required final byte, between 0x40 and 0x7e
    """,
    re.VERBOSE,
)

# The last part of the input which *might* start a CSI escape sequence.
# This matches every possible *incomplete* CSI escape sequence, including
# the empty string for convenience (so that the regex always matches).
csi_suffix = re.compile(
    r"""
    \Z
    | \x1b \Z
    | \x1b \[ [0-?]* [ -/]* \Z
    """,
    re.VERBOSE,
)


class CsiFilterCodec(codecs.Codec):
    """Stateless encoder and decoder which filter out CSI escape sequences."""
    @staticmethod
    def encode(input: str, errors="strict") -> tuple[bytes, int]:
        """Discard csi escape sequences and encode the resulting string in utf-8."""
        consumed = csi_suffix.search(input).start()
        filtered = csi_escape.sub("", input[:consumed])
        output, _ = codecs.utf_8_encode(filtered, errors)
        return output, consumed

    @staticmethod
    def decode(input: bytes, errors="strict") -> tuple[str, int]:
        """Decode a utf-8 string and discard csi escape sequences from the result."""
        unfiltered, consumed = codecs.utf_8_decode(input, errors, False)
        start, end = csi_suffix.search(unfiltered).span()
        # since CSI escape sequences are ASCII,
        # the number of utf-8 bytes is the same as the string length
        consumed -= end - start
        output = csi_escape.sub("", unfiltered[:start])
        return output, consumed


class CsiFilterIncrementalEncoder(codecs.BufferedIncrementalEncoder):
    """Incremental encoder which filters out CSI escape sequences."""
    # The codecs.BufferedIncrementalEncoder super-class implements all the
    # necessary methods as long as we supply a _buffer_encode method.
    def _buffer_encode(self, input: str, errors: str, final: bool) -> tuple[bytes, int]:
        return CsiFilterCodec.encode(input, errors)


class CsiFilterIncrementalDecoder(codecs.BufferedIncrementalDecoder):
    """Incremental decoder which filters out CSI escape sequences."""
    # The codecs.BufferedIncrementalDecoder super-class implements all the
    # necessary methods as long as we supply a _buffer_decode method.
    def _buffer_decode(self, input: bytes, errors: str, final: bool) -> tuple[str, int]:
        return CsiFilterCodec.decode(input, errors)


class CsiFilterStreamReader(CsiFilterCodec, codecs.StreamReader):
    """Stream reader class used by `open(..., 'r', encoding='csi-filter')`."""
    # The codecs.StreamReader super-class implements all the necessary methods
    # as long as we supply a decode method, which the CsiFilterCodec
    # super-class provides.


class CsiFilterStreamWriter(CsiFilterCodec, codecs.StreamWriter):
    """Stream writer class used by `open(..., 'w', encoding='csi-filter')`."""
    # The codecs.StreamWriter super-class implements all the necessary methods
    # as long as we supply an encode method, which the CsiFilterCodec
    # super-class provides.


def register_codec():
    """Register the string "csi-filter" as the name of a usable encoding."""

    @codecs.register
    def lookup(encoding):
        # note that `encoding` is normalized to all lower case letters,
        # with hyphens and spaces converted to underscores
        if encoding == "csi_filter":
            return codecs.CodecInfo(
                name="csi-filter",
                encode=CsiFilterCodec.encode,
                decode=CsiFilterCodec.decode,
                incrementalencoder=CsiFilterIncrementalEncoder,
                incrementaldecoder=CsiFilterIncrementalDecoder,
                streamreader=CsiFilterStreamReader,
                streamwriter=CsiFilterStreamWriter,
            )
