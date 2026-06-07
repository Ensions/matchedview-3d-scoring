# matchedview-3d-scoring


Evaluation notebook for image-to-3D mesh quality, built on Hunyuan3D-2.
The core idea: instead of comparing the input photo against all rendered views of the mesh, the notebook first finds the rendered views whose silhouettes best match the input photo, then runs CLIP, DINOv2, and SigLIP only on those. This isolates mesh quality from viewpoint mismatch noise.
Also includes geometry sanity checks, bootstrap stability analysis, and an HTML report.
Tested on the OpenIllumination dataset. Requires a Colab A100.

Work in progress. The scoring weights and calibration ranges are still being tuned.
