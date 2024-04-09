"""
Dataset registry for fetching with Pooch.

The `registry` dictionary maps file names to checksums used by Pooch to verify
download integrity. To generate checksums for new files, use `openssl md5 <filename>`.
"""

registry = {
    "swo_ecoplot_128x128.zip": "md5:17bb5df154f944e24d1ae465b40eabb0",
    "swo_ecoplot_2048x4096.zip": "md5:c5da200670f1e4426b5f0ad5145790f0",
}
