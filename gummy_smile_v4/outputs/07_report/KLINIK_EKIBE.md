# Klinik ekibe — metni sizden beklenen hakem maddeleri

Kaynak: `docs/Hakem_Yorumları.docx` (hakem mektubunun tam metni, 60 madde). Bu dosyada yalnızca **metni klinik ekibin yazacağı 23 madde** var; teknik analizden cevaplanan 32 madde `RESPONSE_TO_REVIEWERS.md` içinde hazır. Üretim: `scripts/build_rebuttal.py`; elle düzenlemeyin.

Her madde için: hakemin **birebir** sözü, ne gerektiği, ve teknik tarafın şimdiden sağladığı metin/sayılar. Son cümleleri yazarken teknik kısmı olduğu gibi kullanabilirsiniz; sayılar `outputs/` dosyalarından okunuyor ve elle yazılmadı.

## R1-novelty — Technical novelty is incremental and integrative rather than foundational

**Hakem (Reviewer 1):**

> - The works described show a novel application of AI techniques on smile line classification.
> But the technical novelty of the method appears to be incremental and integrative rather than
> foundational.

**Ne gerekiyor:** Introduction and Discussion — the contribution restated: the segmentation is the instrument, the controlled comparison and the evaluated rule layer are the contribution

**Teknik tarafın sağladığı (taslak metin, sayılar çıktı dosyalarından):**

We accept the assessment of the segmentation component and no longer present it as the contribution. Two reviewers say the same thing from different directions, and the revision answers them together: the segmentation is a competent application of existing architectures, not a new method, and it is reported as the instrument the study needs rather than as its result. What the revision does claim, and what is new relative to the group's own previous work, is the controlled architecture comparison — a pre-registered protocol, identical data and budget, three seeds, a decision rule fixed before the runs, with the millimetre measurement as the primary outcome rather than mAP — and the explicit, auditable rule layer whose agreement with blinded clinical assessment is measured rather than assumed. Whether that is sufficient novelty for this journal is the editors' judgement, and the manuscript now states the contribution plainly enough for them to make it.

Kaynaklar: `outputs/08_architecture/RESULTS.md`, `outputs/07_report/MANUSCRIPT_EDITS.md`

## R1-discussion-length — The Discussion is too long and repetitive

**Hakem (Reviewer 1):**

> - The length of Discussion is too long and some content are repeatedly mentioned. The authors
> should revise and shorten them.

**Ne gerekiyor:** Discussion — shortened and restructured around this study's results; repeated literature moved to the Introduction; general conclusions removed

**Teknik tarafın sağladığı (taslak metin, sayılar çıktı dosyalarından):**

Accepted. Three reviewers make the same point — the Discussion is long, repeats the literature and draws conclusions the study did not test — and it is rewritten around this study's own results: the measurement accuracy against the clinical reference, the boundary analysis that explains where the error comes from, the controlled architecture comparison, the learning curve that has not plateaued, and the agreement of the decision layer with independent clinical assessment. Literature that belongs to the rationale moves to the Introduction, the general passages are cut, and the conclusions are restricted to what was measured.

Kaynaklar: `outputs/07_report/MANUSCRIPT_EDITS.md`

## R2-11 — Difference and novelty against the authors' previous study (Ref. 30, J Dent 2026)

**Hakem (Reviewer 2):**

> 11. The authors’ previous study (Ref. [30], Çankaya et al., J Dent, 2026) appears to address
> similar gingival display quantification. The specific differences and novelty of the present
> study should therefore be clarified.

**Ne gerekiyor:** Methods 2.1 (relationship to the earlier study); `outputs/07_report/overlap_with_prior_study.md` [to be generated]

**Teknik tarafın sağladığı (taslak metin, sayılar çıktı dosyalarından):**

Draft from the clinical team (to be confirmed and signed off by them): "Both studies draw on the same source pool of standardised smile photographs acquired at our centre. The previous study (J Dent 2026) used only high-smile-line images (n = 687) for model development, with a single "visible gingiva" class segmented by DeepLabV3+. Of the 1,315 images in the present study, 149 high-smile-line images were also part of the previous model-development set; the remaining 1,166 images (67 high, 303 low and 796 average smile line) were not analysed previously. After removal of duplicate photographs (Section 2.x), 145 of the 1,230 retained images overlap with the previous study. No training, validation or test partition from the previous study was reused: the present data were re-annotated with a two-class (lip, gingiva) instance-segmentation scheme, partitioned anew at the participant level, and used to train a different architecture. The two studies therefore share source images but differ in labelling scheme, model, task (lip–gingiva instance segmentation and threshold-based decision support versus gingiva-only segmentation and regression) and analytical outputs." Technical check: 149 of the 216 high-smile-line images have a reference measurement in the earlier study's measurement file; the earlier partition was not reused (new participant-level split, different label scheme and model).

Kaynaklar: `docs/Klinik_ekip_kararlari_15Eylul.md`, `outputs/01_data/OZET.md`

## R3-General-1 — Rewrite the introduction with a specific aim and references for all statements

**Hakem (Reviewer 3):**

> 1. The authors should rewrite the introduction to include a specific aim and provide references
> for all statements.

**Ne gerekiyor:** Introduction — rewritten: one aim, a reference for every statement, the gap restated as the decision layer rather than the measurement

**Teknik tarafın sağladığı (taslak metin, sayılar çıktı dosyalarından):**

Accepted. The Introduction is rewritten with one stated aim and a reference for every claim it makes. What the technical side supplies for it: the gap this study can legitimately claim is not that gingival display has never been quantified automatically — the group's own J Dent 2026 study did that — but that no published system converts the measurement into an explicit, auditable rule set and then measures how far that layer agrees with clinicians. The Introduction will say so in those words, and the sentences that generalise about 'AI-based image analysis' are replaced by the specific prior work with citations.

Kaynaklar: `outputs/07_report/MANUSCRIPT_EDITS.md`

## R3-General-2 — Clarify the materials and methods

**Hakem (Reviewer 3):**

> 2. The authors should clarify the materials and methods.

**Ne gerekiyor:** Methods 2.1-2.9 — restructured; calibration, measurement geometry, partition and evaluation protocol each given their own subsection

**Teknik tarafın sağladığı (taslak metin, sayılar çıktı dosyalarından):**

Accepted, and most of it is already rewritten. The Methods now state the design and the recruitment (clinical team), and, from the analysis side: the participant-level partition and its exact counts, the annotation procedure, the pixel-to-millimetre calibration (item Methods 6), the measurement geometry and the single fixed estimator with its sensitivity analysis, the training configuration of the reported model, the evaluation protocol with the test set used once, and the statistical methods with their software. Each of those is a numbered subsection so that a reader can follow the pipeline end to end.

Kaynaklar: `outputs/07_report/MANUSCRIPT_EDITS.md`

## R3-General-4 — Rewrite the discussion on the study outcomes; draw relevant rather than general conclusions

**Hakem (Reviewer 3):**

> 4. The authors should rewrite the discussion based on the study outcomes and draw relevant
> rather than general conclusions.

**Ne gerekiyor:** Discussion — shortened and restructured around this study's results; repeated literature moved to the Introduction; general conclusions removed

**Teknik tarafın sağladığı (taslak metin, sayılar çıktı dosyalarından):**

Accepted. Three reviewers make the same point — the Discussion is long, repeats the literature and draws conclusions the study did not test — and it is rewritten around this study's own results: the measurement accuracy against the clinical reference, the boundary analysis that explains where the error comes from, the controlled architecture comparison, the learning curve that has not plateaued, and the agreement of the decision layer with independent clinical assessment. Literature that belongs to the rationale moves to the Introduction, the general passages are cut, and the conclusions are restricted to what was measured.

Kaynaklar: `outputs/07_report/MANUSCRIPT_EDITS.md`

## R3-General-5 — Improve the language of the manuscript

**Hakem (Reviewer 3):**

> 5. The authors should improve the language of the manuscript to enhance readability

**Ne gerekiyor:** Whole manuscript — professional language editing

**Teknik tarafın sağladığı (taslak metin, sayılar çıktı dosyalarından):**

Accepted. The manuscript is going through professional language editing before resubmission, and the certificate will accompany it.

## R3-Title-1 — The title promises more than was validated

**Hakem (Reviewer 3):**

> The title includes more than what was validated in the study: ‘Toward etiological interpretation
> and treatment planning’. In this study the clinical validity was not evaluated. '

**Ne gerekiyor:** Title / subtitle — rewritten so the claim matches what was validated

**Teknik tarafın sağladığı (taslak metin, sayılar çıktı dosyalarından):**

Accepted. The subtitle promises what the study does not validate, and it is changed: the revision proposes 'measurement accuracy and a rule-based framework for etiological interpretation' in place of 'Toward etiological interpretation and treatment planning'. The final wording is the clinical team's, but it will not contain a claim of clinical validation of treatment planning.

Kaynaklar: `outputs/07_report/MANUSCRIPT_EDITS.md`

## R3-Intro-1 — A reference is missing from the paragraph describing the state of the literature

**Hakem (Reviewer 3):**

> 1. The authors should add a reference or references to the paragraph containing this sentence
> ((In the literature, artificial intelligence–based image analysis studies have predominantly
> focused....))

**Ne gerekiyor:** Introduction — rewritten: one aim, a reference for every statement, the gap restated as the decision layer rather than the measurement

**Teknik tarafın sağladığı (taslak metin, sayılar çıktı dosyalarından):**

Accepted. The Introduction is rewritten with one stated aim and a reference for every claim it makes. What the technical side supplies for it: the gap this study can legitimately claim is not that gingival display has never been quantified automatically — the group's own J Dent 2026 study did that — but that no published system converts the measurement into an explicit, auditable rule set and then measures how far that layer agrees with clinicians. The Introduction will say so in those words, and the sentences that generalise about 'AI-based image analysis' are replaced by the specific prior work with citations.

Kaynaklar: `outputs/07_report/MANUSCRIPT_EDITS.md`

## R3-Intro-2 — The stated scientific gap is general and does not match what the study evaluates

**Hakem (Reviewer 3):**

> 2. The statement ((In the literature, artificial intelligence–based image analysis studies have
> predominantly focused....)) which describes the scientific gap is general. From this sentence I
> understand that there is only a gap in the clinical application (etiological/treatment
> decisions) of the AI-model. But the study did not evaluate the clinical performance of the
> AI-model. Is there also a scientific gap in the: the accuracy of automated anatomical
> segmentation or the validity of millimetric measurements performed by the AI model?

**Ne gerekiyor:** Introduction — the scientific gap restated and narrowed to what the study tests

**Teknik tarafın sağladığı (taslak metin, sayılar çıktı dosyalarından):**

The reviewer reads the sentence correctly and the reading exposes a real problem: as written, the gap is only in the clinical application, which is the part the study does not evaluate. The revision states the gap where the study actually contributes and says which of the two questions the reviewer raises is open. Automated segmentation of gingiva and lip is not an open question in general; the millimetre validity of the measurement derived from it is reported here against a clinical reference (MAE 0.52 mm, RMSE 0.73 mm, r = 0.892, ICC(2,1) 0.877 [0.805, 0.919], bias +0.27 mm [+0.16, +0.38], 95 % LoA -1.07 to 1.60 mm; Table 1 class agreement 76 %, linear-weighted κ 0.72 [0.63, 0.80]), and the open question the study addresses is whether an explicit rule layer on top of that measurement agrees with clinical judgement.

Kaynaklar: `outputs/07_report/MANUSCRIPT_EDITS.md`

## R3-Intro-3 — Several objectives are presented at once; identify one primary objective

**Hakem (Reviewer 3):**

> 3. The final section of the introduction contains several objectives that are not clearly
> presented. Identify one primary objective and present the additional objectives separately.

**Ne gerekiyor:** Introduction, final paragraph — one primary objective, secondary objectives listed separately

**Teknik tarafın sağladığı (taslak metin, sayılar çıktı dosyalarından):**

Accepted. The revision names one primary objective — to quantify gingival display from a frontal smile photograph and report its accuracy in millimetres against a clinical reference — and lists the secondary objectives separately: the controlled comparison of segmentation architectures, and the agreement of the rule-based etiological/treatment layer with independent clinical assessment.

Kaynaklar: `outputs/07_report/MANUSCRIPT_EDITS.md`

## R3-Methods-1 — Study design, retrospective/prospective, recruitment process and dates

**Hakem (Reviewer 3):**

> 1. The study design should be described more precisely: Is this a cross-sectional study? Is it
> retrospective or prospective? The recruitment process and dates should also be mentioned.

**Ne gerekiyor:** Methods 2.1 — design, retrospective/prospective, recruitment process and dates

**Teknik tarafın sağladığı (taslak metin, sayılar çıktı dosyalarından):**

Accepted; this belongs to the clinical team and the wording is being supplied. The design is a retrospective, cross-sectional analysis of photographs and clinical records collected in a single centre under ethics approval E-77082166-604.01-881629, and the recruitment window and the consecutive/selective nature of the sampling are being stated explicitly, because the reviewer's later question about selection bias (Discussion 3) cannot be answered without them. From the analysis side, what can already be stated is the flow from the export to the analysed set: 1315 exported images, 1230 analysed after the exclusions, partitioned at participant level into 846 / 192 / 192.

Kaynaklar: `outputs/07_report/tables/dataset_counts.md`

## R3-Methods-2 — Inclusion criteria are not given, only exclusion criteria

**Hakem (Reviewer 3):**

> 2. What are the inclusion criteria. The study mentioned only the exclusion criteria.

**Ne gerekiyor:** Methods 2.1 — inclusion criteria added beside the exclusion criteria

**Teknik tarafın sağladığı (taslak metin, sayılar çıktı dosyalarından):**

Correct, and it is a real omission: only exclusion criteria were given. The inclusion criteria are being written by the clinical team. The analysis-side filters that act on top of them are already documented and will be stated in the same place, because they determine which images enter which analysis: an image enters the segmentation training set if it has a usable annotation, and it enters the millimetre analysis only if it is a high smile line with a clinical reference measurement (n = 145). Duplicate photographs of the same participant were reduced to one image per participant before partitioning.

Kaynaklar: `outputs/07_report/tables/dataset_counts.md`

## R3-Methods-9 — Was a systematic search used to synthesise the evidence behind Table 1?

**Hakem (Reviewer 3):**

> 9. Table 1 is used as an evidence-based clinical reference for the study. Did you conduct a
> systematic search to synthesize the evidence from references 14–23? Could you describe the
> method used?

**Ne gerekiyor:** Methods — Table 1 described as a narrative, literature-derived synthesis with the source of each band; no claim of a systematic search

**Teknik tarafın sağladığı (taslak metin, sayılar çıktı dosyalarından):**

No, a systematic search was not performed, and the manuscript should not have implied otherwise. Table 1 is a narrative synthesis of the thresholds used in references 14-23, assembled by the clinical authors from the literature they work with. The revision says exactly that: it describes the table as a literature-derived, non-systematic synthesis, states the criterion by which each threshold was taken, and lists the source of every band, so that a reader can see which numbers are widely used and which are one group's convention. The clinical team will supply the description of how the references were gathered. If the editors prefer, the table can instead be presented as the pre-specified rule set this study evaluates, with its provenance given and no claim of evidence synthesis attached to it.

## R3-Methods-10 — Table 1 has clinical deficiencies (a: is < 4 mm a problem; b: aetiology from millimetres alone; c: short upper lip needs a lip measurement)

**Hakem (Reviewer 3):**

> 10. Table 1 has some clinical deficiencies: a. Is a gingival display of less than 4 mm a
> problem? Is a 1 mm gingival display, which is less than 4 mm, a problem, and should it be
> treated? b. I cannot understand how the aetiology of gingival display can be diagnosed based
> only on the amount of gingival display, without clinical assessment or other assessments, such
> as cephalometric analysis. c. The diagnosis of a short upper lip requires measurement of the
> length of upper lip. Upper-lip measurements may differ according to gender, age, and ethnicity.

**Ne gerekiyor:** Methods 2.8 and the Table 1 caption — the rule set described as a generator of candidate etiologies, not a diagnosis; overlapping bands and their combined labels made explicit; Limitations — lip length, cephalometry and periodontal findings are not inputs; Discussion — the expert agreement as the measure of this limit

**Teknik tarafın sağladığı (taslak metin, sayılar çıktı dosyalarından):**

Two reviewers make this objection — Reviewer 3 in Methods 10 (a, b and c) and Reviewer 4 in his second fundamental problem — and they are right on the substance. The clinical framing is the clinical team's to write; what the technical side can state, and what the design already does, is this. **The system does not diagnose.** It measures gingival display in millimetres and applies a published threshold table to produce one or more *candidate* etiologies with the treatments associated with them in the literature. The output field is named `treatment_alternatives`, not 'treatment'. **Overlapping bands are reported as overlaps, not resolved.** The bands of Table 1 overlap by construction (E1 below 4 mm, E2 from 3 to 6, E3 from 4 to 8, E4 above 8), so a 5 mm display returns the combined label E2-E3 and both sets of candidates rather than a single answer. The engine has no metadata-based tie-breaking and never picks one etiology from a measurement alone — which is precisely the reviewers' point, built into the rule set rather than argued against it. **On 10a specifically:** a display below 4 mm is not asserted to be a problem. The rule returns a category for an observed display; it does not assert an indication, and a value of 0 mm returns `NO_VISIBLE_GINGIVA` rather than a class. **On 10c and the short upper lip:** we agree that the diagnosis requires a lip measurement and that lip length varies with sex, age and ethnicity. The system does not measure lip length and therefore cannot diagnose a short lip; where the band admits that etiology it is listed as a candidate to be confirmed clinically. The revision says so in the Methods, in the Table 1 caption and in the Limitations. **And this is exactly what the expert study measures.** Three clinicians assign the etiology from their own clinical judgement, blinded to the table and to each other; the linear-weighted κ between their majority and the rule output is the quantity that says how far a millimetre-only rule can go. Whatever that number turns out to be, it is the honest measure of this limitation, and it is reported either way.

Kaynaklar: `docs/Uzman_degerlendirme_protokolu.md`, `outputs/07_report/tables/expert_agreement.md`

## R3-Methods-11 — Abbreviations such as YOLO must be expanded at first mention

**Hakem (Reviewer 3):**

> 11. Abbreviations such as YOLO should be explained when first time mentioned in the text.

**Ne gerekiyor:** Whole manuscript — abbreviations expanded at first mention

**Teknik tarafın sağladığı (taslak metin, sayılar çıktı dosyalarından):**

Accepted. Every abbreviation is expanded at first mention in the revision — YOLO (You Only Look Once), RF-DETR (Receptive Field enhanced Detection Transformer), mAP (mean average precision), IoU (intersection over union), MAE, RMSE, ICC and LoA — and a definitions list is added where the journal allows one.

Kaynaklar: `outputs/07_report/MANUSCRIPT_EDITS.md`

## R3-Results-1 — Demographics not reported; ethnicity and pigmentation of skin and gingiva may affect segmentation

**Hakem (Reviewer 3):**

> 1. The study did not report the demographic characteristics of the sample. The mean age should
> be reported with the standard deviation. The ethnicity of the patients may play a role, as some
> patients have pigmentation of the skin and gingiva. Could this affect the segmentation of the
> lips and gingiva and, consequently, the performance of the AI model?

**Ne gerekiyor:** Limitations — new sentence (pigmentation and ethnicity not recorded); Discussion — nature of the segmentation error

**Teknik tarafın sağladığı (taslak metin, sayılar çıktı dosyalarından):**

This is a fair point and we cannot answer it with these data: neither ethnicity nor gingival or skin pigmentation was recorded for the cohort, so no subgroup analysis is possible. We state this as a limitation rather than speculate. What we can report is where the segmentation error actually lies, which does not look like a pigmentation effect: the error is a systematic displacement of one boundary rather than a random failure of the mask. On the 145 reference images the upper, lip-side gingiva edge is accurate (MAE 0.28 mm, bias +0.02 mm) while the lower, festooned margin is placed consistently too low (bias +0.07 mm), and the same shift appears in the fold models and in the final model alike (+0.04 mm on the test-set high-smile-line images). A pigmentation-driven failure would be expected to vary between participants rather than to be constant. This is evidence about the nature of the error, not about pigmentation itself; a prospective study recording phenotype would be required to answer the reviewer's question properly, and we say so. [CLINICAL — the limitation sentence is to be finalised by the clinical team.]

Kaynaklar: `outputs/09_final_rfdetr/boundary_by_set.md`, `outputs/09_final_rfdetr/error_decomposition.md`

## R3-Discussion-1 — The discussion repeats literature that belongs in the introduction

**Hakem (Reviewer 3):**

> 1. The discussion repeats findings from the literature that should be presented only in the
> introduction.

**Ne gerekiyor:** Discussion — shortened and restructured around this study's results; repeated literature moved to the Introduction; general conclusions removed

**Teknik tarafın sağladığı (taslak metin, sayılar çıktı dosyalarından):**

Accepted. Three reviewers make the same point — the Discussion is long, repeats the literature and draws conclusions the study did not test — and it is rewritten around this study's own results: the measurement accuracy against the clinical reference, the boundary analysis that explains where the error comes from, the controlled architecture comparison, the learning curve that has not plateaued, and the agreement of the decision layer with independent clinical assessment. Literature that belongs to the rationale moves to the Introduction, the general passages are cut, and the conclusions are restricted to what was measured.

Kaynaklar: `outputs/07_report/MANUSCRIPT_EDITS.md`

## R3-Discussion-2 — The discussion should focus on this study's results and discuss the model's validity

**Hakem (Reviewer 3):**

> 2. The discussion should focus on the results of this study, support them with evidence from
> previous studies, compare the findings, and discuss the validity of the AI model.

**Ne gerekiyor:** Discussion — shortened and restructured around this study's results; repeated literature moved to the Introduction; general conclusions removed

**Teknik tarafın sağladığı (taslak metin, sayılar çıktı dosyalarından):**

Accepted. Three reviewers make the same point — the Discussion is long, repeats the literature and draws conclusions the study did not test — and it is rewritten around this study's own results: the measurement accuracy against the clinical reference, the boundary analysis that explains where the error comes from, the controlled architecture comparison, the learning curve that has not plateaued, and the agreement of the decision layer with independent clinical assessment. Literature that belongs to the rationale moves to the Introduction, the general passages are cut, and the conclusions are restricted to what was measured.

Kaynaklar: `outputs/07_report/MANUSCRIPT_EDITS.md`

## R3-Discussion-4 — A conclusion appears twice: at the end of the discussion and as its own section

**Hakem (Reviewer 3):**

> 4. A conclusion is presented at the end of the discussion, followed by a separate Conclusion
> section.

**Ne gerekiyor:** Discussion — the closing conclusion paragraph removed; one Conclusion section kept

**Teknik tarafın sağladığı (taslak metin, sayılar çıktı dosyalarından):**

Accepted; this is an editing error. The revision keeps one Conclusion section and removes the concluding paragraph at the end of the Discussion.

Kaynaklar: `outputs/07_report/MANUSCRIPT_EDITS.md`

## R3-Discussion-5 — Conclusion must not state the model is clinically validated

**Hakem (Reviewer 3):**

> 5. The conclusion should not that the AI model has been clinically validated or is as this
> aspect was not evaluated in the study.

**Ne gerekiyor:** [Conclusions — rewritten]; [Abstract, Conclusions and Clinical Significance — rewritten]; Discussion limitation paragraph

**Teknik tarafın sağladığı (taslak metin, sayılar çıktı dosyalarından):**

We accept this without reservation and have removed every statement that implies clinical validation of the decision layer, in the Conclusions, the Abstract and the Clinical Significance. The manuscript now separates three claims: (i) the segmentation model performs as reported on an independent test set; (ii) the measurement agrees with the clinical reference to MAE 0.52 mm on out-of-fold predictions; (iii) the etiology-treatment layer is a transparent application of published thresholds whose agreement with clinical judgement is assessed in an agreement study, not a validation of diagnostic accuracy. The Conclusion states explicitly that the framework has not been validated as a clinical decision tool, that the E4 branch has no case in this cohort (maximum mean gingival display 7.53 mm) and that prospective, multi-centre clinical validation is required before use. [CLINICAL — final wording from the clinical team.]

Kaynaklar: `outputs/07_report/MANUSCRIPT_EDITS.md`

## R4-2 — Overlap with the authors' J Dent 2026 study, same ethics number; originality and redundant publication

**Hakem (Reviewer 4):**

> The originality of the segmentation component is low to moderate. As early as 2022, automatic
> segmentation of teeth, gums, and facial structures for digital smile design was published. In
> 2024, the Journal of Dentistry published a “smile index” based on peri-oral segmentation to
> automate smile classification. More importantly, the authors themselves published the
> aforementioned study on gingival exposure segmentation with clinical validation just a few
> months ago. They explicitly acknowledge that the novelty of the new manuscript does not lie in
> improving measurement accuracy, but rather in linking that measurement to a system of clinical
> interpretation. Another issue needs clarification. The previous study used the same ethics
> approval number, E-77082166-604.01-881629, which appears in this manuscript. The previously
> published article reports 1,748 photographs from the same project. This strongly suggests
> significant overlap in the study population and possibly in the images between the two studies.
> This is not necessarily a problem if it is a clearly distinct secondary analysis. Still, the
> authors must state exactly how many participants/images appear in both articles, which data are
> new, which training/validation/test sets are reused, and why the present study does not
> constitute redundant publication. Until this is clarified, it is impossible to assess the actual
> originality properly.

**Ne gerekiyor:** Methods 2.1 (relationship to the earlier study); `outputs/07_report/overlap_with_prior_study.md` [to be generated]

**Teknik tarafın sağladığı (taslak metin, sayılar çıktı dosyalarından):**

Draft from the clinical team (to be confirmed and signed off by them): "Both studies draw on the same source pool of standardised smile photographs acquired at our centre. The previous study (J Dent 2026) used only high-smile-line images (n = 687) for model development, with a single "visible gingiva" class segmented by DeepLabV3+. Of the 1,315 images in the present study, 149 high-smile-line images were also part of the previous model-development set; the remaining 1,166 images (67 high, 303 low and 796 average smile line) were not analysed previously. After removal of duplicate photographs (Section 2.x), 145 of the 1,230 retained images overlap with the previous study. No training, validation or test partition from the previous study was reused: the present data were re-annotated with a two-class (lip, gingiva) instance-segmentation scheme, partitioned anew at the participant level, and used to train a different architecture. The two studies therefore share source images but differ in labelling scheme, model, task (lip–gingiva instance segmentation and threshold-based decision support versus gingiva-only segmentation and regression) and analytical outputs." Technical check: 149 of the 216 high-smile-line images have a reference measurement in the earlier study's measurement file; the earlier partition was not reused (new participant-level split, different label scheme and model).

Kaynaklar: `docs/Klinik_ekip_kararlari_15Eylul.md`, `outputs/01_data/OZET.md`

## R4-9 — The etiology of a gummy smile cannot be derived from millimetres alone; this contradicts the logic of Table 1

**Hakem (Reviewer 4):**

> The second problem is even more fundamental: the etiology of a gummy smile cannot be derived
> solely from the number of millimeters of visible gum tissue. The authors themselves acknowledge
> that a definitive diagnosis requires cephalometry, lip morphology, and a periodontal
> examination. This statement partially contradicts the logic of Table 1. A patient with 5 mm of
> exposure may present with lip hypermobility, maxillary vertical excess, altered passive
> eruption, dentoalveolar extrusion, or combinations thereof. Similarly, diagnosing a “short upper
> lip” requires measuring the lip; diagnosing VME requires facial/skeletal assessment; diagnosing
> altered passive eruption requires evaluating coronal dimensions and the dentogingival
> relationship; and establishing an indication for orthognathic surgery cannot be based solely on
> exposure exceeding 8 mm. The matrix can help generate a differential diagnosis, but it should
> not be used as a treatment tool without independent clinical validation.

**Ne gerekiyor:** Methods 2.8 and the Table 1 caption — the rule set described as a generator of candidate etiologies, not a diagnosis; overlapping bands and their combined labels made explicit; Limitations — lip length, cephalometry and periodontal findings are not inputs; Discussion — the expert agreement as the measure of this limit

**Teknik tarafın sağladığı (taslak metin, sayılar çıktı dosyalarından):**

Two reviewers make this objection — Reviewer 3 in Methods 10 (a, b and c) and Reviewer 4 in his second fundamental problem — and they are right on the substance. The clinical framing is the clinical team's to write; what the technical side can state, and what the design already does, is this. **The system does not diagnose.** It measures gingival display in millimetres and applies a published threshold table to produce one or more *candidate* etiologies with the treatments associated with them in the literature. The output field is named `treatment_alternatives`, not 'treatment'. **Overlapping bands are reported as overlaps, not resolved.** The bands of Table 1 overlap by construction (E1 below 4 mm, E2 from 3 to 6, E3 from 4 to 8, E4 above 8), so a 5 mm display returns the combined label E2-E3 and both sets of candidates rather than a single answer. The engine has no metadata-based tie-breaking and never picks one etiology from a measurement alone — which is precisely the reviewers' point, built into the rule set rather than argued against it. **On 10a specifically:** a display below 4 mm is not asserted to be a problem. The rule returns a category for an observed display; it does not assert an indication, and a value of 0 mm returns `NO_VISIBLE_GINGIVA` rather than a class. **On 10c and the short upper lip:** we agree that the diagnosis requires a lip measurement and that lip length varies with sex, age and ethnicity. The system does not measure lip length and therefore cannot diagnose a short lip; where the band admits that etiology it is listed as a candidate to be confirmed clinically. The revision says so in the Methods, in the Table 1 caption and in the Limitations. **And this is exactly what the expert study measures.** Three clinicians assign the etiology from their own clinical judgement, blinded to the table and to each other; the linear-weighted κ between their majority and the rule output is the quantity that says how far a millimetre-only rule can go. Whatever that number turns out to be, it is the honest measure of this limitation, and it is reported either way.

Kaynaklar: `docs/Uzman_degerlendirme_protokolu.md`, `outputs/07_report/tables/expert_agreement.md`

---

## Ayrıca: uzman formları beklenen 5 madde

Bunların metni teknik tarafta hazır; eksik olan yalnızca üç uzmanın doldurduğu formlardan gelecek sayılar (`run_expert_analysis.py` → `outputs/04_expert/manuscript_numbers.md`). Formlar geldiğinde cevaplar kendiliğinden tamamlanır.

| madde | hakem | konu | ne bekleniyor |
|---|---|---|---|
| R2-7 | Reviewer 2 | Single examiner; no inter-rater reliability, so a possible ground-truth bias | uzman formlarından gelecek sayılar |
| R3-General-3 | Reviewer 3 | Narrow the manuscript to a validated segmentation/measurement study, or compare with clinical assessment | uzman formlarından gelecek sayılar |
| R3-Abstract-1 | Reviewer 3 | Abstract reports no millimetric accuracy and no validation of the etiological/treatment categories | uzman formlarından gelecek sayılar |
| R4-1 | Reviewer 4 | The etiology-treatment module is not validated; no clinical gold standard, no agreement with clinicians | uzman formlarından gelecek sayılar |
| R4-1b | Reviewer 4 | No cohort in which clinicians independently establish the cause and the plan; no sensitivity, specificity or kappa | uzman formlarından gelecek sayılar |
