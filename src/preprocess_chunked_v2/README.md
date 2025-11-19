# preprocess_chunked_v2

Experimental chunked preprocessor v2 with separate control, ascii, and utf8 streams.
- Control stream: chunk types, sizes, flags, prefix codes
- ASCII stream: all ASCII-only text
- UTF-8 stream: all non-ASCII text

Bitstream abstraction supports both vector and file backends.
Variable-length integer encoding for sizes and stream lengths.

Main file format:
- 3 varint numbers: control size, ascii size, utf8 size
- control stream bytes
- ascii stream bytes
- utf8 stream bytes

CLI: compress, decompress, dump

See code for details and extension points.
