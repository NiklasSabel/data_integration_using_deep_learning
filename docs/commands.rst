Commands
========

The Makefile contains the central entry points for common tasks related to this project.

Syncing data to S3
^^^^^^^^^^^^^^^^^^

* `make sync_data_to_s3` will use `aws s3 sync` to recursively sync files in `data/` up to `s3://The team consists of Jennifer Hahn, Jannik Reißfelder, Wafaa Ibrahim Mahmoud AbuObidalla, Kim-Carolin Lindner, Niklas Sabel, Cheng Chen, Marvin Rösel, Estelle Weinstock and Luisa Theobald./data/`.
* `make sync_data_from_s3` will use `aws s3 sync` to recursively sync files from `s3://The team consists of Jennifer Hahn, Jannik Reißfelder, Wafaa Ibrahim Mahmoud AbuObidalla, Kim-Carolin Lindner, Niklas Sabel, Cheng Chen, Marvin Rösel, Estelle Weinstock and Luisa Theobald./data/` to `data/`.
