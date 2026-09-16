# Numbers for the manuscript (value, 95 % CI, n)

| metric | value | 95 % CI | n |
|---|---|---|---|
| Model vs expert majority, linear-weighted κ (primary: all images, OOF, strict) | 0.549 | [0.388, 0.688] | 133 |
| … unweighted κ | 0.499 | [0.332, 0.643] | 133 |
| … observed agreement | 0.797 | [0.729, 0.865] | 133 |
| … PABAK | 0.729 | [0.639, 0.820] | 133 |
| Model vs expert majority, linear-weighted κ (secondary: fixed test subset, strict) | 0.471 | [-0.145, 0.836] | 27 |
| … corrected model values (offset_px +0, secondary), linear-weighted κ (primary set, strict) | 0.549 | [0.388, 0.688] | 133 |
| … lenient scoring, linear-weighted κ (primary set) | 0.717 | [0.567, 0.832] | 133 |
| Inter-expert Fleiss κ (class) | 0.130 | [0.036, 0.218] | 142 |
| Inter-expert ICC(2,1), image-mean mm (3 experts) | 0.991 | [0.988, 0.993] | 142 |
| Inter-expert ICC(2,k), image-mean mm (3 experts) | 0.997 | [0.996, 0.998] | 142 |
| ICC(2,1), 3 experts + clinical reference | 0.993 | [0.991, 0.995] | 142 |
| Model vs expert mean, ICC(2,1) mm — uncorrected, PRIMARY | 0.868 | [0.819, 0.904] | 145 |
| … bias (model − expert mean), mm — uncorrected, PRIMARY | +0.16 | [+0.03, +0.28] | 145 |
| Model vs expert mean, ICC(2,1) mm — corrected, secondary | 0.868 | [0.819, 0.904] | 145 |
| … bias (model − expert mean), mm — corrected, secondary | +0.16 | [+0.03, +0.28] | 145 |
| Model vs clinical reference, ICC(2,1) mm — uncorrected, PRIMARY | 0.868 | [0.819, 0.904] | 145 |
| … bias (model − clinical reference), mm — uncorrected, PRIMARY | +0.15 | [+0.03, +0.27] | 145 |
| Model vs clinical reference, ICC(2,1) mm — corrected, secondary | 0.868 | [0.819, 0.904] | 145 |
| … bias (model − clinical reference), mm — corrected, secondary | +0.15 | [+0.03, +0.27] | 145 |
| Intra-expert κ (linear), expert 1 | 0.291 | [-0.154, 0.635] | 20 |
| Intra-expert ICC(2,1) image mm, expert 1 | 0.993 | [0.983, 0.997] | 20 |
| Intra-expert κ (linear), expert 2 | 0.101 | [-0.245, 0.402] | 19 |
| Intra-expert ICC(2,1) image mm, expert 2 | 0.991 | [0.978, 0.997] | 19 |
| Intra-expert κ (linear), expert 3 | 0.346 | [-0.058, 0.667] | 20 |
| Intra-expert ICC(2,1) image mm, expert 3 | 0.992 | [0.981, 0.997] | 20 |
| Expert scale agreement ICC(2,1), px/mm | 0.356 | [0.249, 0.465] | 134 |
| Tooth-level bias (mixed model intercept), mm — uncorrected (PRIMARY) | +0.154 | [+0.034, +0.275] | 870 |

E4 has no reference case in this dataset (per_class.csv shows n = 0); the E4/T4 branch is not validated.

## Pre-specification note
The protocol originally named the fixed test subset (n = 29) as the primary set for the class-agreement analysis and all 145 images as secondary. This was reversed **before any real expert form was available**, on the basis of the synthetic dry run: with n = 29 the bootstrap 95 % CI of the linear-weighted κ spanned roughly −0.15 to 0.84, i.e. the primary estimate would have been uninformative. The 5-fold out-of-fold predictions are equally unbiased (every image is predicted by a model that never saw it or its same-patient twin), so the primary set is now all 145 reference images with OOF predicted masks; the fixed test subset (final model) is reported as secondary. The same rule applies in Stage 6. In this run the model table is: GT masks / synthetic dry run.
