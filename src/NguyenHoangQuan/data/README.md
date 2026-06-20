# Data folder

Place metadata and audio files here.

Expected metadata columns:

```csv
filename,primary_label
XC12345.ogg,sp1
XC67890.ogg,sp2
```

Audio can be flat:

```text
data/audio/XC12345.ogg
```

or class-subfolder based:

```text
data/audio/sp1/XC12345.ogg
```

For class order, create:

```text
data/classes.txt
```

one class label per line.
