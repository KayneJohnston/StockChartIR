# The working paper

`lifecycle_asset_allocation.pdf` is a full working-paper writeup of the entire
project: abstract, data, methodology, the baseline replication, all eight
extensions, discussion, limitations and four appendices.

```bash
python paper/build_paper.py                    # -> paper/lifecycle_asset_allocation.pdf
python paper/build_paper.py --out /tmp/x.pdf   # somewhere else
```

The build is a two-pass ReportLab render (the second pass resolves the table
of contents' page numbers). It is deliberately *not* self-contained:

* **Prose** lives in `content.py`.
* **Every number** in that prose is resolved at build time by `facts.py`,
  which reads `results/tables/*.csv` and `config.yaml`. Nothing is typed by
  hand, so a pipeline rerun that changed a result changes the paper rather
  than leaving it silently contradicting the evidence. A missing table fails
  the build loudly.
* **Figures** are the same PNGs the pipeline writes to `results/figures/`.
* `style.py` holds the typography and the flowable helpers, including a
  `missing_glyphs` check that the build runs over all prose — ReportLab draws
  an unmapped codepoint as a black box with no warning, which is easy to ship
  by accident in a document this full of Greek.

Run `python main.py` first; the paper cannot be built from an empty
`results/` directory.
