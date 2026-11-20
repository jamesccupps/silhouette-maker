# Silhouette Maker: notes for working on it

A local app that turns side-on photos into silhouette cut files (Cricut SVG, PNG, print pages, STL). A Flask server
on 127.0.0.1 and one page. The README says what it does; this file is what to keep in mind when changing or
releasing it.

## The rule that comes first

The photos are of children. Nothing may send a photo, a silhouette or a name anywhere: no analytics, no web fonts,
no CDN scripts, no update check that carries data. `data/` (the working copies) and `.test-data/` are never
committed. Screenshots, tests and issue examples use the drawn profiles in `samples/` (`tools/make_samples.py`),
never real photos.

## Commands

- Run: `Start Silhouette Maker.bat`, or `.venv\Scripts\python.exe app.py` (port 8777, data in `data/`).
- A test copy next to a running one: `python app.py --no-browser --port 8778 --data .test-data`.
- Tests: `.venv\Scripts\python.exe -m pytest -q`. They draw their own masks; no cut-out model is downloaded.
  GitHub runs them on Windows and Linux with Python 3.10 and 3.13: 3.10 is the oldest supported, so no newer
  library behaviour may be assumed (0.5.0 broke restore there: zipfile could not read an upload's stream).
- Setup: `setup.bat` (picks the graphics runtime), `setup.bat nvidia`, `setup.bat directml`.

## How it is built

- `engine.py`: cut-out (rembg), cleanup, outline, lettering as outlines (fontTools), SVG/PNG, print pages.
  Bump `engine.REV` when `outline()` changes its result: outlines are cached in each photo's `outline.json`.
- `model3d.py`: STL files (Shapely), in millimetres. Every solid must stay closed: the tests count open edges.
- `app.py`: the server. It answers only requests addressed to its own address and coming from its own page.
- `static/`: the page. No build step. Never name a top-level function after a browser global (`open` broke
  `window.open`).
- The onnxruntime builds cannot be installed side by side: `onnxruntime-gpu[cuda,cudnn]` (NVIDIA),
  `onnxruntime-directml`, plain `onnxruntime`. `requirements.txt` pulls none on Windows; `setup.bat` chooses.
  The *Best* model needs the graphics card; on the CPU it runs out of memory on Windows.
- Speck-filling happens once at cut-out time (`clean_cutout`), never in `outline()`: it would undo erasing.

## Looking at changes

Judge the page from full-size screenshots of real states (a headless browser at 1500 × 900 over the test copy),
with lettering switched on and every section opened: the 3D picture failed for curved Georgia lettering in 0.5.0
and only a screenshot showed it.

## Releasing

1. Bump `__version__` in `app.py`; update the README and, when the screens changed, the pictures in `docs/`
   (the project page shows the same files).
2. Commit, push, wait for the Tests workflow to pass on all four jobs.
3. Tag `vX.Y.Z` (annotated) and push the tag.
4. Make the zip from the tag: `git archive --format=zip --prefix=Silhouette-Maker/ -o Silhouette-Maker.zip vX.Y.Z`.
5. Publish a GitHub release titled `Silhouette Maker X.Y.Z` with the notes and the zip attached under exactly the
   name `Silhouette-Maker.zip`: the README and the page link to `releases/latest/download/Silhouette-Maker.zip`.

The project page is `site/index.html`, published by `.github/workflows/pages.yml` when `site/` or `docs/*.png`
change. Keep its SEO tags when editing (title up to 60 characters, description up to 160, canonical, `og:*`,
theme-color, image sizes) and its "More tools" section.

## Style

Plain words in the page, the README and messages: short sentences, no jargon, say what a control does. Commits are
made as `James <148652101+jamesccupps@users.noreply.github.com>`.
