# Evaluation protocol

Claim policy: "matches or beats a hosted baseline" is allowed only with all of:
- baseline version pinned (not `latest`), same items, instructions, criteria and K.
- Media items: baseline is the text model + ASR/OCR/caption.
- Zero-shot held-out schemas and fine-tuned reported as separate rows.
- Raw NLL/Brier/ECE with no test-domain temperature, plus post-T row with T fitted on a disjoint split.
- Per K (include K>20, Banking77 full), per modality, per language.
- Paired cluster bootstrap by state, pre-registered margin, frozen test split.
- Matched-concurrency latency; evaluator independent of vendors.
- Contamination flag (`in_training`) on every row.

Metrics: accuracy, macro-F1, NLL, Brier, adaptive ECE, acc@coverage, coverage at 5% error, zero-probability failure rate, score MAE, option-order flip rate, complement/partition/implication coherence (`opendecision.coherence`), counterfactual sensitivity.

Known pitfalls from prior art: Laya fine-tuned ECE 0.213 vs headline 0.081 (post-T on another run); Baseline numbers in competing READMEs are third-party and unpaired; zero-shot Laya 0.362 < majority baseline 0.461.
