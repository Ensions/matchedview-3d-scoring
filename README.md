# MatchedView 3D Scoring

**Viewpoint-aware quality scoring for image-to-3D generation, built on [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2).**

Generate a mesh from a photo, then score how well that mesh actually matches the photo — without letting viewpoint mismatch pollute the number. Runs end-to-end in a single Colab notebook and produces per-image scores, reliability flags, diagnostic images, and a sortable HTML report.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ensions/matchedview-3d-scoring/blob/main/main.ipynb)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-work%20in%20progress-orange)

> **Status:** work in progress. The scoring weights and calibration ranges are still being tuned, so treat absolute scores as provisional — relative rankings and reliability flags are the more trustworthy signals right now.

---

## The problem

The standard way to score image-to-3D output is to render the generated mesh from N viewpoints and compare each render against the input photo with a semantic metric like CLIP. But the input photo is **one unknown viewpoint**, while the renders cover the whole sphere. Averaging similarity over all views mixes two unrelated things:

1. How good the mesh actually is.
2. How lucky you got with viewpoint overlap.

A great mesh photographed from an unusual angle scores poorly; a mediocre blob can score decently because blobs look the same from everywhere. The score stops measuring mesh quality.

## The fix: matched-view scoring

Instead of comparing against all renders, the evaluator first **finds the renders that were taken from (approximately) the same viewpoint as the photo**, then runs the semantic metrics only on those:

1. **Render 24 views** of the mesh (3 elevations × 8 azimuths) with pyrender.
2. **Silhouette-match** each render against the input photo using multi-scale IoU (crop-normalized masks, scales 0.85 / 1.0 / 1.15 to tolerate proportion drift).
3. **Pick the top K = 3 views by IoU** — the "matched views."
4. **Score CLIP, DINOv2, and SigLIP similarity on the matched views only.** This is the primary, viewpoint-isolated quality signal.
5. Keep the **full 24-view set** for consistency, blob-detection, and diversity checks.

Three independent semantic backbones are used because they fail differently: CLIP (ViT-H-14, LAION-2B) is color-biased, DINOv2 (ViT-L/14) is shape-aware, and SigLIP (ViT-B-16-384) is a strong third opinion for image–image similarity. When all three agree, the score is trustworthy; when they diverge, the evaluator says so. DINOv2 and SigLIP are optional and degrade gracefully if they fail to load.

## What goes into the final score

| Component | What it measures | Weight |
|---|---|---|
| Matched-view CLIP / DINOv2 / SigLIP | Semantic similarity to the input, viewpoint-isolated | 72% (visual), split across available signals |
| Matched-view silhouette IoU | Calibration-free shape match | folded into visual score |
| Geometry sanity | Face count, connected components, watertightness, aspect ratio, hull ratio (penalizes both blobs and paper-thin shells) | 28% |
| Consistency penalty | Some views scoring drastically worse than others | up to −25 |
| Blob penalty | All renders nearly identical → likely a featureless blob | up to −20 |
| Fragility penalty | Score swings across bootstrap subsets, or weak matched viewpoint | up to −18 |

Raw cosine similarities are mapped to a 0–100 scale with **per-model calibration ranges**, since untextured renders vs. real photos occupy very different cosine bands for each backbone (e.g. CLIP ≈ 0.18–0.42, DINOv2 ≈ 0.40–0.72, SigLIP ≈ 0.28–0.58).

### Reliability flags

Every image gets a `high` / `medium` / `low` reliability flag with explicit reasons, based on:

- **Cross-signal agreement** — do CLIP, DINOv2, SigLIP, and IoU tell the same story?
- **Bootstrap stability** — the score is recomputed on 8 random 12-view subsets; high standard deviation means the metric is fragile.
- **Matched-view quality** — if no rendered viewpoint resembles the input silhouette, the whole premise is weak and the flag drops.
- **Geometry health and blob detection.**

Interpretation:

- `high` → trust the score; signals agree and it's stable.
- `medium` → the ranking is probably right; the absolute number may be noisy.
- `low` → don't trust the number; open the spotlight image and judge by eye.

## Outputs

Everything is written to the output directory (default `/content/hunyuan_eval_outputs`):

- `hunyuan_multisignal_scores.json` — full per-image breakdown: final score, every sub-score, per-view raw values, view metadata, rank percentile, and review reasons. Stream-written after each image, so a mid-batch crash loses nothing.
- `report.html` — sortable table of all results with color-coded reliability and links to the diagnostic images.
- `<image>_spotlight.jpg` — the input photo side-by-side with its top-3 matched renders. The single most useful diagnostic: you can see at a glance whether the match makes sense.
- `<image>_preview.jpg` — 24-view grid with per-view CLIP/DINO/SigLIP/IoU overlaid and matched views outlined in red.
- `<image>.obj` — the generated mesh, for inspection in any 3D viewer.

## Quickstart (Google Colab)

Requires a GPU runtime — an **A100** is recommended (Hunyuan3D-2 mesh generation plus three vision backbones is heavy).

The notebook is three cells, run in order:

1. **Cell 1 — Setup.** Clones Hunyuan3D-2, installs system EGL/Mesa libraries and Python dependencies (pyrender, trimesh, open_clip). The apt installs must complete *before* pyrender is imported, which is why order matters.
2. **Cell 2 — Evaluator.** Defines the full pipeline and loads it into the session. Includes a renderer self-test (renders a known icosphere) that fails loudly if Colab's EGL context is broken, instead of silently scoring everything 0.
3. **Cell 3 — Run.** Prompts for the number of images, lets you upload them (`png`/`jpg`/`jpeg`/`webp`), generates a mesh per image, scores it, and prints ranked results with spotlight previews inline.

If the renderer self-test fails or every view comes back empty: **Runtime → Restart runtime**, then re-run Cells 1 → 2 → 3 in order.

### Tweakables

Key constants at the top of Cell 2:

```python
RENDER_SIZE = 384          # render resolution
STEPS = 30                 # Hunyuan3D inference steps
ELEVATIONS = [0.0, 0.4, 0.8]
N_AZIMUTHS = 8             # 3 × 8 = 24 views
MATCHED_K = 3              # top-K silhouette-matched views to score on
IOU_SCALES = [0.85, 1.0, 1.15]
BOOTSTRAP_N = 8            # bootstrap resamples
BOOTSTRAP_SUBSET = 12      # views per resample
SEED = 2025                # fixed seed for reproducible mesh generation
```

`evaluate_images(..., resume=True)` skips images already present in the JSON, so interrupted batches can be resumed.

## Design choices worth knowing

- **Trimmed means and max-aware aggregation** make per-view statistics robust to pathological angles and self-occlusion outliers.
- **Multi-scale IoU** avoids unfairly punishing meshes with the right shape but slightly wrong proportions.
- **Hull-ratio scoring penalizes both extremes** — near-1 (sphere/cube blob) and near-0 (paper-thin or broken geometry).
- **Rank percentiles** are included in the JSON as a calibration-free signal for comparing within a batch.
- Tested on the [OpenIllumination](https://oppo-us-research.github.io/OpenIllumination/) dataset.

## Limitations

- Calibration ranges were tuned for **untextured (shape-only) renders vs. real photos**; textured-render comparison would need different ranges.
- Matched-view selection assumes the input silhouette is extractable — heavy occlusion or busy backgrounds degrade the anchor (this is flagged as low reliability rather than hidden).
- Absolute scores are still being calibrated; cross-batch comparisons should use reliability flags and rank percentiles.

## License

MIT — see [LICENSE](LICENSE).
