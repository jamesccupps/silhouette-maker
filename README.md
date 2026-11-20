# Silhouette Maker

Turn side-on photos into silhouette cut files for a Cricut, entirely on your
own computer. Made for a daycare ornament project: photograph each child in
profile, cut the silhouette from black vinyl or paper, put it on a wooden
ornament.

![The gallery: one card per child](docs/gallery.png)

![The editor: photo, cut line, neckline and ornament preview](docs/editor.png)

*The pictures above use the drawn sample profiles in `samples/`, not real children.*

## Photos never leave the computer

- The cut-out is done by a model running on your PC. Nothing is uploaded.
- The app is a page served from `127.0.0.1` and refuses requests from any other
  computer or web site.
- The only download is the cut-out model itself, fetched once on first use
  (about 170 MB for *Fast*, about 1 GB for *Best*). After that it works offline.
- Working copies of the photos are ordinary files in the `data` folder next to
  the app. **Delete** in the app removes them.

## What it does

- **Automatic cut-out** of each photo, stood upright if it was taken sideways
  or upside down. JPEG, PNG, WebP and iPhone HEIC photos all work.
- **Touch-up**: Add and Erase brushes with undo and zoom; turn and straighten;
  a grid to line things up.
- **Neckline**: three dots shape the bottom of the bust.
- **Made for cutting**: smoothing, and a "thinnest detail kept" setting in
  millimetres at the finished size, so hair wisps too thin to cut and weed are
  dropped.
- **Sized in inches**, with an exact preview on your ornament: circle, oval,
  square, rectangle, hexagon, heart or star, in the colours you are using.
- **Gallery**: search, mark finished, and download one SVG per child, a zip of
  all, or everyone packed onto mat-sized sheets.
- **Files for Cricut Design Space**: SVG at true size, optionally with a box or
  the ornament's shape as its own layer, and PNG.
- **Lettering**: the child's name and a second line such as the year, straight
  under the silhouette or curved along the edge. Long names wrap and shrink to
  stay on the ornament.
- **Proof sheet**: print everyone at real size on plain paper before cutting.
- **Privacy tools**: delete the original photos once you are done and keep only
  the silhouettes; back up and restore everything as one file.

## Install (Windows)

You need Windows 10 or 11 and [Python](https://www.python.org/downloads/) 3.10
to 3.13.

1. Download this repository (**Code → Download ZIP**) and unzip it, or clone it.
2. Double-click **Start Silhouette Maker.bat**. The first run sets itself up
   (a few minutes), then a page opens in your browser.
3. To stop the app, close the black window.

Double-click **Add desktop shortcut.bat** once to get a Silhouette Maker icon on
your Desktop.

**Best** cut-out quality uses the graphics card (any DirectX 12 card; about
15 seconds a photo on an RTX 3060). If it can't run on your PC, the app says so
and uses **Fast**, which runs on any CPU in about a second and is rougher
around hair.

The app looks at the computer when it starts (graphics card, its memory, the
PC's memory) and says under **Add photos…** which method suits it. When you add
photos it asks **Best or Fast** before any work starts, showing its
recommendation and how long the batch will take, timed from real cut-outs on
your computer. Tick "Use this from now on" to stop being asked; **Cut-out
method** on the left changes it back.

While it is cutting out, the *Best* model uses about 10 GB of graphics memory.
It is let go 90 seconds after the last cut-out, so the app is light the rest of
the time; the next photo after a pause takes about 10 seconds longer.

### Linux and macOS

Not tested. It should run with:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

There the *Best* model runs on the CPU and needs roughly 16 GB of free memory;
otherwise the app falls back to *Fast*.

## Use

Hold the mouse over any button or setting for a short explanation, and use
**Help** (bottom left) for the steps and the keyboard shortcuts.

1. Drop photos on the page or use **Add photos…**. Try the ones in `samples/`
   first. Each is cut out and lands in the **gallery**. Everything saves as you
   work; there is no Save button.
2. Click a card to open it. **← Gallery**, Esc or the browser's Back closes it.
3. Tidy up if needed:
   - **Add / Erase** (B / E) paint the shape in or away. Scroll to zoom, hold
     Space to move, **Fit** (F) shows the whole photo again.
   - **Cut out again** reruns the automatic cut-out with the model chosen next
     to it, if the first try missed.
   - **Undo** (Ctrl+Z) and **Redo** (Ctrl+Y) step through up to 100 changes,
     brush strokes and settings alike. **History** lists every step since you
     opened the photo; click one to go back to it. The list starts fresh each
     time a photo is opened.
   - **Reset to how it started** (in History) puts back the first automatic
     cut-out and the starting settings, at any time, even days later.
   - **⟲ Turn / Turn ⟳** and **Straighten** fix a sideways or tilted photo;
     **Grid** (G) shows lines to straighten against.
   - **Neckline** (N): drag the three dots.
   - **Smoothing** rounds the outline; **Thinnest detail kept** drops what is
     too thin to cut.
   - The orange line is exactly what will be cut.
4. Set the **Silhouette height** and the ornament, tick **Finished**. **Same for
   everyone** gives every child the same silhouette height, so the ornaments
   match; photos added later get it too.
5. **Download SVG for Cricut**. From the gallery, **Download all SVGs** gives a
   zip and **Sheet of all** packs the children onto sheets for the mat chosen
   beside it: 12 × 12, 12 × 24, Joy 4.5 × 12, or a letter page. Design Space
   may shrink a file taller than 11.5 in, so check the size after uploading a
   12 × 24 sheet.

### The ornament and lettering

Under **Ornament and size**, choose the ornament's shape, size and colour, and
the colour of your vinyl. The picture there is drawn by the same code that
writes the files, so it shows exactly what you will get, and it says so when the
silhouette is too big for the ornament.

Under **Lettering**, tick **Put the name on it** and, if you like, type a
**Second line** (the year, a class name). The name and the second line each have
their own font and size, and every font installed on the computer is offered,
with a few that cut well suggested at the top; install a new font in Windows,
restart the app, and it is in the list. The letters can sit straight under the
silhouette or, on a round ornament, curve along the bottom edge. A long name
wraps onto two lines and then shrinks, so it never runs off the ornament. Photos
straight from a camera (IMG_2041…) come in without a name; type one in the
**Name** box.

The letters are written into the SVG as outlines, not as text, because Design
Space ignores text in an uploaded SVG. They arrive as their own layer, so you
can cut them, draw them with a pen, or delete them. Small cut letters are hard
to weed, so try the proof sheet first.

**Behind the silhouette** (under **Files**) adds a plain box or the ornament's
own shape to the files as a separate layer: a size guide you can delete, or a
backing you can cut.

### Before you cut: the proof sheet

**Proof sheet** in the gallery opens a page with every silhouette at its real
size, sitting on a dashed circle the size of your ornament, with names. Print it
at 100% ("Actual size", not "Fit"); the black bar at the top should measure
exactly 2 inches. Check likeness and fit on paper before using vinyl.

### When you are finished

- **Delete the photo, keep the silhouette** (in the editor), or tick children in
  the gallery and use **Delete photos only**. The original photo is removed from
  the computer; the silhouette, its settings and the cut files stay. It cannot be
  cut out again afterwards.
- **Back up** (bottom left) saves everything as one zip: photos, silhouettes,
  settings. It contains the photos, so keep it somewhere private. **Restore**
  adds the children from a backup file and leaves the ones already here alone.

### In Cricut Design Space

**Upload → the SVG → add to canvas.** The file is written in the units Design
Space assumes, so it should arrive at the size shown in the app. Check the
first one; if it is off, type the width into the **W** box with the lock on.
On a sheet, each shape carries the child's name in the layers panel.

### Photo tips

- Child looking straight sideways, camera level with the head.
- A plain, bright wall behind, and a little space around the head.
- Hair off the face, so forehead, nose, lips and chin show.
- Neck and shoulders in the frame, so there is room for the neckline.

## For developers

```bash
.venv\Scripts\python -m pip install pytest
.venv\Scripts\python -m pytest -q tests
```

- `engine.py`: cut-out, cleanup, outline and SVG/PNG writing.
- `app.py`: the local server. `--port N` and `--data FOLDER` run a second copy
  for testing without touching real photos.
- `static/`: the page. No build step.
- `tools/make_samples.py` draws the sample profiles; `tools/make_icon.py` the icon.

Please never put a real child's photo in an issue, a pull request or a test.

## Credits

- [rembg](https://github.com/danielgatis/rembg) (MIT) runs the cut-out models.
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) (MIT), the *Best* model.
- [U²-Net](https://github.com/xuebinqin/U-2-Net) (Apache 2.0), the *Fast* model.
- [fontTools](https://github.com/fonttools/fonttools) (MIT) reads the letter shapes.
- [ONNX Runtime](https://onnxruntime.ai/), [OpenCV](https://opencv.org/),
  [scikit-image](https://scikit-image.org/), [Pillow](https://python-pillow.org/),
  [NumPy](https://numpy.org/) and [Flask](https://flask.palletsprojects.com/).

The model files are downloaded from those projects and are under their own
terms. Cricut and Design Space are trademarks of Cricut, Inc.; this project is
not affiliated with Cricut.

## License

[MIT](LICENSE)
