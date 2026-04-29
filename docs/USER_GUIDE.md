# SegOid User Guide

A friendly walkthrough for researchers using SegOid to analyze spheroid images. This guide covers the two-step workflow: running the SegOid program on your images, then analyzing the results in your browser.

No coding required.

---

## What SegOid does

SegOid is a program that finds spheroids in your brightfield microscopy images and measures them automatically. You give it a folder of images; it gives you back:

- **Predicted masks** — pictures showing exactly which pixels are spheroids.
- **A measurements file** — a spreadsheet with one row per spheroid, listing things like area, circularity, and (if you set it up) fluorescence intensity from your live/dead stains.
- **A review video** — a slideshow of every image with the predictions overlaid, so you can quickly spot mistakes.

After SegOid produces these, you can use a second tool (a notebook that runs in your web browser) to filter out bad detections, group results by condition, and export a clean spreadsheet ready for GraphPad or Prism.

---

## The full workflow at a glance

```
   Your microscope                 SegOid program                 Browser notebook
        |                                |                                |
   Stitch & export                Open folder of images          Upload measurements
   TIFF images                          ↓                                ↓
        ──────────────►        Click Run, wait                Edit a few settings
                                       ↓                                ↓
                              metrics.csv + masks            Multi-tab Excel ready
                              + review video.mp4              for GraphPad / Prism
```

You only deal with two interfaces: the SegOid app on your computer, and a notebook in your browser.

---

## Before you start

You will need:

- **A Windows computer** with the SegOid app installed (your maintainer will give you the `SegOid.exe` file).
- **A folder of TIFF images** to analyze — the stitched, exported images from your microscope.
- **A Google account** (for the browser notebook step). If your lab restricts what you can upload to cloud services, check with your IT person — the notebook only sees the small numbers file (no images), but it does run on Google's servers.

---

## Step 1: Run SegOid on your images

This step finds the spheroids and produces the measurements file.

### 1a. Open the app

Double-click `SegOid.exe`. A window opens.

> [Screenshot: SegOid main window]

### 1b. Pick your input folder

Click the **Browse** button next to "Input image folder" and navigate to the folder containing your stitched TIFF images.

**Tip:** put all the images you want analyzed into the same folder. SegOid will process every TIFF file it finds.

### 1c. Pick where the results should go

Click **Browse** next to "Output folder" and pick or create a folder for the results. SegOid will fill it with the prediction masks, measurements file, and (optionally) the review video.

### 1d. *(Optional but very useful)* Parse filenames into columns

If your images are named in a consistent pattern — for example, `control_lowdose_image1.tif`, `drug_highdose_image2.tif` — SegOid can pull the parts of the name into separate columns in the measurements file. This makes the next step (analysis) much easier because you can group by condition, dose, etc.

How to use it:

1. Tick the checkbox **"Parse filename into fields (split by '_')"**.
2. A text box appears. Type the labels you want to use, separated by commas, in the order they appear in your filenames.

**Example.** If your files are named like `control_lowdose_image1.tif`, type:

```
condition, dose, image
```

When SegOid is done, the measurements file will have new columns called `condition`, `dose`, and `image` filled in for every spheroid.

**Important:** the number of labels you type must match the number of underscore-separated parts in your filenames. If your files are `control_image1.tif` (two parts), type two labels (`condition, image`), not three.

### 1e. *(Optional)* Detect fluorescence markers

If you also captured live/dead images (e.g., a green channel and a red channel) at the same fields of view, SegOid can measure the average green/red intensity inside each spheroid it detects.

Set up your folder like this:

- For each brightfield image (e.g., `well_A1.tif`), put the matching channel files in the same folder, named `well_A1_green.tif` and `well_A1_red.tif`.
- The fluorescence files **must be single-channel grayscale TIFFs** (the standard quantitative export from your microscope, not RGB-coloured display images). If your microscope exports RGB composites, ask your microscope manager how to export the raw intensity per channel instead.

Then in SegOid:

1. Tick **"Detect fluorescence markers (companion images <base>_<marker>.tif)"**.
2. In the text box that appears, type the marker names you used, separated by commas. For example:

```
green, red
```

SegOid will then add two new columns to the measurements file: `mean_intensity_green` and `mean_intensity_red`. Each row will hold the average intensity of that channel inside that spheroid.

If a brightfield image doesn't have a matching companion file, that cell will be blank ("NA"). SegOid will print a warning message in the log telling you how many were missing.

### 1f. Click Run

Hit the **Run Inference** button. The log area at the bottom will scroll through messages as SegOid processes each image. A typical batch of 10–30 images takes a few minutes.

You can click **Cancel** if you need to stop. When the run finishes, the **Open Output Folder** button becomes active.

### 1g. What you'll find in the output folder

When SegOid is done, your output folder will contain:

| File | What it is |
|------|------------|
| `<image_name>_pred_mask.tif` | The predicted spheroid mask for each input image. Black = background, white = spheroid. |
| `<image_name>_pred_prob.tif` | The "confidence map" showing how sure SegOid was about each pixel (mostly for advanced users). |
| `metrics.csv` | **The big one.** One row per detected spheroid, with all measurements (area, circularity, fluorescence intensity, plus your filename parsing columns). |
| `summary_statistics.csv` | Quick averages of the measurements (mean area, mean circularity, etc.) — handy for a sanity check. |
| `prediction_review.mp4` | *(if you ticked the video option)* A slideshow video showing each image with the predicted mask overlaid. Watch this to spot any predictions that look obviously wrong. |

The next step is to take the `metrics.csv` and turn it into a clean Excel ready for plotting.

---

## Step 2: Analyze the results in the browser notebook

This step doesn't need anything installed on your computer — it runs in Google's web browser.

### 2a. Open the notebook

In the SegOid project's GitHub page, look for the **"Open In Colab"** badge in the `colab/` folder. Click it. Your browser will open the notebook.

(Direct link: search the GitHub page for the file `colab/post_segmentation.ipynb` and click "Open in Colab" at the top.)

> [Screenshot: Colab open with the SegOid notebook]

### 2b. Run the install step

The first cell installs SegOid into the temporary notebook. Click the play button (▶) at the left edge of the first code cell. Wait for the green checkmark — about 1 minute the first time.

### 2c. Upload your measurements file

Click the play button on the **upload** cell. A file picker appears. Choose the `metrics.csv` file from your SegOid output folder. Wait for the upload to finish.

### 2d. Edit the settings block

Scroll down to the cell that looks like this:

```
# === EDIT THESE ===
group_by = ['condition']
circularity_min = 0.5
area_min_px = 10000
area_max_px = 50000
area_min_um2 = None
area_max_um2 = None
mad_max = None
output_filename = 'post_seg.xlsx'
# ====================
```

Edit the values for your run. Each line is one setting:

- **`group_by`** — which columns to group results by. Use the column names you set up with "Parse filename into fields" in Step 1. Examples:
  - `['condition']` — one tab per condition.
  - `['condition', 'dose']` — one tab per condition + dose combination.
  - `['image']` — one tab per source image (the default).
- **`circularity_min`** — drops spheroids that are too irregular. Typical: `0.5`. Set to `None` to keep all.
- **`area_min_px`** / **`area_max_px`** — drops spheroids smaller/larger than these pixel-area thresholds. Adjust to match what you consider "real" spheroids in your assay. Set either to `None` to skip that filter.
- **`area_min_um2`** / **`area_max_um2`** — same idea but in physical units (square micrometers). Use these instead of the px versions if you set a pixel size in SegOid.
- **`mad_max`** — drops outliers based on robust statistics. Set to `None` initially. If you want to flag unusually-sized spheroids, try `3` (drops anything more than 3 robust standard deviations from the median area in its group).
- **`output_filename`** — what the resulting Excel file will be called.

A setting of `None` means "don't filter on this." A number means "apply this threshold."

### 2e. Run the analysis

Click play on the **run the analysis** cell. It takes a few seconds.

### 2f. Download the Excel file

Click play on the **download** cell. Your browser downloads the Excel.

### 2g. What's inside the Excel

Open it. You'll see several tabs:

| Tab | What's in it |
|-----|--------------|
| One tab per group (e.g., `control`, `drug`) | The spheroids in that group that **passed** all your filters. Ready to copy into GraphPad. |
| `all_with_filter_status` | Every spheroid, including filtered-out ones. The `passed_filter` column tells you which were kept. Use this to audit what your filters did. |
| `group_stats` | Quick summary statistics (n, mean, median, IQR, MAD) for each group. Good for a sanity check. |
| `graphpad_long` | All the kept spheroids in "long format" (one row per spheroid, with a `group` column). This format imports cleanly into GraphPad Prism if you prefer that workflow over the per-group tabs. |

You can re-run with different filter settings as many times as you like — just edit the settings block and run cells 2d, 2e, 2f again.

---

## End-to-end example

Sarah has 32 wells: 4 cell lines × 8 drug doses. Her microscope produces one stitched TIFF per well, named like:

```
A549_1uM_well1.tif        H1299_10uM_well3.tif
A549_5uM_well2.tif        Normal_DMSO_well1.tif
... (32 files total)
```

She also captured live/dead at the end:

```
A549_1uM_well1_green.tif       (calcein, alive)
A549_1uM_well1_red.tif         (PI, dead)
... (one green + one red per brightfield)
```

**Step 1 — Run SegOid.**

She drops all 96 files (32 brightfield + 32 green + 32 red) into one folder. In the SegOid app:

- Input folder: that folder.
- Output folder: a new empty folder.
- Tick **Parse filename into fields**, fill in: `cell_line, dose, well`.
- Tick **Detect fluorescence markers**, fill in: `green, red`.
- Click **Run Inference**.

After ~5 minutes, her output folder contains 32 mask files, a `metrics.csv` with about 200 rows (one per detected spheroid), and a review video. The metrics file has columns `cell_line`, `dose`, `well`, `mean_intensity_green`, `mean_intensity_red` filled in for every row.

**Step 2 — Analyze.**

She opens the Colab notebook, uploads `metrics.csv`, and edits the settings:

```
group_by = ['cell_line', 'dose']
circularity_min = 0.5
area_min_px = 10000
area_max_px = 50000
```

She runs the analysis and downloads the Excel. It has 32 tabs (one per cell_line × dose combination), each containing only the well-formed, correctly-sized spheroids and their measurements. She copies the area and intensity columns into GraphPad and makes her plots.

The whole process: ~5 minutes of microscope-image processing + ~30 seconds of analysis + her usual GraphPad time. The manual sorting/filtering she used to do in Excel is gone.

---

## Troubleshooting

**"No image files found in input folder"**
SegOid only looks for files ending in `.tif` or `.tiff`. Check that your files have one of these extensions (lowercase or uppercase both work). Subfolders are not searched — every image must be directly in the folder you select.

**The notebook says "Grouping columns not found in metrics CSV"**
The columns you put in `group_by` don't exist in your measurements file. Most often this means you forgot to tick **"Parse filename into fields"** when running SegOid in Step 1. Re-run SegOid with that option enabled, then re-upload the new `metrics.csv` to the notebook.

**A `mean_intensity_green` (or red) column is empty for some rows**
That brightfield image didn't have a matching `_green.tif` (or `_red.tif`) companion file in the input folder. Check your folder — the companion filename must exactly match the brightfield filename plus the marker suffix. Capitalization is flexible, but spelling and underscores must match.

**SegOid rejects my fluorescence file with "not single-channel grayscale"**
Your microscope exported the file as a colour image (e.g., a green-on-black RGB image meant for display) rather than a single-channel grayscale image with raw intensity values. Re-export the channel from your microscope software, picking the option for raw / single-channel / 16-bit grayscale, not the coloured / RGB / merged option. If you're not sure how, ask your microscope manager.

**I get too many or too few spheroids detected**
SegOid's mistakes are visible in the review video. If it's missing real spheroids or detecting things that aren't spheroids, the post-segmentation filters in the notebook (circularity, area) catch most of the bad detections. If you still see lots of mistakes, send the maintainer an email — collecting examples helps improve the model. See `docs/CONTRIBUTING_TRAINING_DATA.md`.

**The notebook is slow / Colab disconnected**
Colab gives free users a few hours of compute per session. For a typical metrics file (a few thousand spheroids), the analysis takes seconds — disconnects are usually because the notebook was idle too long. Just refresh, reconnect, and re-run from the top.

**I don't have a Google account / can't use Colab**
Ask your maintainer for the local Python version of the analysis tool (`post_seg_analyze`). It does the exact same thing without uploading anything.

---

## Where to get help

- For SegOid bugs or feature requests: open an issue on the GitHub repository, or email the maintainer directly.
- To contribute corrected masks (when SegOid gets predictions wrong): see `docs/CONTRIBUTING_TRAINING_DATA.md`.
- For technical details on how SegOid works internally: see `README.md`.
