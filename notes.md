Attribution methods:
-Saliency (baseline, gradient of the output class score with respect to the input image pixels, noisy - focus on edges, cheap to compute)
-Integrated Gradients (it creates a sequence of images. It starts with a "baseline" image (usually a pure black image, meaning "zero information") and slowly blends it step-by-step (e.g., 50 steps) into the original input image. It calculates the gradients at every single step and averages them, compute-heavy)
-Occlusion(RISE alternative, a black square slides across the image, then measures how much the confidence dropped)

Sensitivity "metric" - (if a pixel changes the output, it gets a non-zero score)

Experiments plan:
-Calibrated Model + topn% pixels removal
-Calibrated Model + buckets removal
-Base Model + buckets removal
-Base Model + buckets guided inpainting

Metrics:
-FID + OTDD (when datasets A - otiginal, B - masked, C - after inpainting prepared)


Experiments:


Ideas to implement and test:
1 - Patch-based Deletion, który usuwa spójne bloki (16x16) zgodne z fizycznym podziałem tokenów w analizowanej architekturze Vit.
2 - Superpixel Deletion (np. SLIC z biblioteki skimage) instead of bucket...
3 - No success with io-painting, so try to replicate papers results (dog with green field)
4 - 