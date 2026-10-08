# ULW-Research Synthesis: OpenDecision, a non-LLM multimodal typed-decision engine
Members: 7 (+ lead) · Waves: 3 · Excursions: 1 (user throughput steering) · Debate rounds: 2 · Verifications: 1 (toy CPU) · Evidence level: mostly abstract/README-level; see Gaps.

## Executive summary
Laya (Apache-2.0) is the open reference: encoder (ModernBERT-large / mmBERT-base) + 2-layer head scoring one [MASK] marker per option, one forward pass for all questions, REINFORCE-style "RLCD" with strictly proper scoring rules. [nar-heads, eval-repo]
Evidence says: (1) the RL term is not what makes it work: with labels, supervised proper-score loss matches or beats RLCD (LAVOIR 2609.30706; eve-rlcd SFT .817 vs RLCD .808); a toy run here confirms the bandit reward r=c-p_a is an unbiased estimator of the half-Brier gradient, while the exact supervised gradient has zero variance [verify-exact-expectation.md]. (2) Calibration and coherence are different axes: hosted decision models miss P(X)+P(not X)=1 by 0.064 on average (2609.33209); calibration fixes leave structural violations unchanged (2607.19367). So OpenDecision ships a coherence battery plus consistency training. (3) Laya's weak spots to beat: 77-label accuracy (.425 vs .870 for the best hosted model), shortcut answers on changed questions (98-99.5% training answer, 2609.33689), overconfident outside English, no media, Python HTTP stack ~10-30k tok/s DERIVED.
Design that survives the debate: non-LLM, encoder-only, encode-state-once + late-interaction question/label head (Perceiver IO style), small per-modality encoders pooled to fixed latents, supervised proper loss + coherence loss + counterfactual-question augmentation, label-free-ish calibration via per-type temperature plus conformal gate. Claim level: parity with hosted baselines only for in-domain fine-tuned low-cardinality text; zero-shot breadth, label-free calibration and multimodal calibration are goals to be PROVEN by a paired protocol, not facts.

## Findings by theme
### Architecture (non-LLM, parallel)
- Backbone <=~300M compute params. Text: ModernBERT-base/large or mmBERT [MEASURED: ModernBERT-large ~50k tok/s on RTX 4090, 2412.13663]. Image: SigLIP2-B NaFlex (~86-92M, 64-196 tokens). Audio: Whisper-base/small encoder cropped to real length (50 tok/s) or CLAP/BEATs (check licence). Video: image encoder on 4-8 frames + temporal pooling; V-JEPA2-L (326M, 2048 tok/clip) only as an optional heavy tier. Each modality pooled to 16-64 latents (Perceiver resampler). 4-6 layer non-causal fusion. Heads: per-question late-interaction cross-attention over cached state, label embeddings precomputed (removes option tokens from budget), softmax (choice), ordinal (score, RPS), sigmoid (noul). [ASSUMED design; Perceiver IO 2107.14795; DETR 2005.12872]
- Question-conditioning: bidirectional encoders cannot reuse a prefix when questions come first, so state is encoded once question-independently, questions are light heads. [DERIVED/design]
- Option-order invariance via shared position ids + masked cross-option attention (Laya parallel_layout) [MEASURED code read].
### Throughput and latency
- FLOPs ~ 2*params*tokens. At 300k tok/s: 86M=51 TFLOPs, 300M=180, 1B=600 [DERIVED, H200 ~989 bf16 peak from memory]. Ceiling ~660M at 40% MFU (optimistic). Attention cost kills it above ~8k context unless local/sliding (ModernBERT alternating 128-window + global). Impossible: 300k tok/s at 32k context; frozen so400m+Whisper-turbo+V-JEPA2-L inside budget.
- "Token" must be defined per modality: SLAs for text tok/s, images/s, audio-seconds/s, video frames/s. Latency boundary = server-side with media already on host; GPU decode (nvJPEG/NVDEC/DALI), [UNVERIFIED rates].
- Laya independent study: H100 NVL ORT-TRT FP16 174.7 decisions/s at p99<=130 ms, eager 93; software bound. Needs Rust router + token-budget batching + CUDA graphs per (batch,seq) bucket + one replica per GPU, no MIG. 200 ms is not the hard part (compute for 2k tokens ~7 ms); throughput and CPU decode are.
- Packed/padding-free execution gives ~2x over eager on 5090 (packed-encoders 0.1.0); no Hopper numbers anywhere. MUST be measured: repo ships bench script.
### Training
- Pretrain/CPT: CLM->MLM (25/75, no LR decay between stages) beats pure MLM for text at 610M (2507.00994, read); untested on vision-language. Staged data mix, specialist data late (SmolLM2). Vision: SigLIP2 multi-loss, MobileCLIP2 offline cached teacher outputs (cheap KD).
- Distillation: Apache teachers only (Qwen2.5-VL-7B, Qwen3-VL-8B Apache; 72B and Qwen2.5-Omni-3B are NOT). Calibrate the teacher before KD (2508.20224); no label smoothing (1906.02629). Judge ceiling: student cannot exceed teacher agreement; eval items must not come from the same judge family.
- Finetuning: supervised proper loss (log for training, Brier for eval/selection; rule choice matters, 2608.28482). LoRA is overconfident, so recalibrate after. Counterfactual question augmentation against shortcutting (2609.33689). Coherence/consistency loss (2404.12843) is non-proper in aggregate, so ablate with NLL/Brier.
- RL: reserved for bandit/outcome-only feedback (r=c-p_a, leave-one-out baseline, eve-rlcd). RLVR with binary reward overconfident (.991 conf, ECE .213). Not the main path.
### Calibration, coherence, uncertainty
- Metrics: acc, macro-F1, NLL, Brier, adaptive/debiased ECE (binned ECE biased, 2203.07835), acc@coverage, coverage@5% error, zero-probability failure rate, order-flip rate, complement-sum/partition/implication coherence battery, per-language/per-modality buckets, OOD-confidence.
- Temperature per (type, K) on a disjoint labelled split; conformal (weighted for drift, 2202.13415) as abstain/escalate gate, never advertised as calibrated probability.
### Evaluation protocol (from skeptic)
Pinned baseline version, same items/instructions/K, text baseline + ASR/caption for media, zero-shot held-out schemas reported separately from fine-tuned, raw (no test-domain T) and post-T rows, per-K/per-modality, K>20, paired cluster bootstrap with pre-registered margin, frozen test split, matched-concurrency latency, contamination flags (in_training) carried in every row.
### Repo / wire
Keep Laya wire (POST /v1/systemone, /batch, /health; answers{q:{type,choice|score|noul,probabilities,confidence,answer_confidence}}, usage, routing), add media refs. Layout to copy: package + docs (mkdocs) + examples + research/results JSON + thresholds.json regression gate + doc-pinning test. Licence Apache-2.0; ship dataset download scripts and DATASETS.md ledger.

## Verified claims
Code: bandit estimator unbiased for half-Brier gradient: CONFIRMED (verify-exact-expectation.md). Non-code cleared (>=2 domains or primary + counter): Laya architecture (code read, common.py); ModernBERT 4090 throughput (paper); CLM->MLM result (paper read); licences of listed encoders (HF API). Everything else is MEDIUM (abstract-level) and flagged.

## Contradictions
- Laya README ECE .081 vs HF card ECE .213 (post-T on other run vs fine-tuned). Resolution: not comparable; per-protocol rows.
- Laya arXiv 2510.01237 cited for RLCD is an unrelated routing paper; no RLCD paper located.
- Laya .766: in-domain fine-tuned; zero-shot Laya .362 < majority .461.

## Gaps
No measured H100/H200 encoder tok/s (MLPerf BERT-large anchor not fetched); multimodal token budgets mostly unverified; multimodal typed-decision benchmarks/licences unverified; 2609.37647/2609.35342/2610.00346 evaluations unread; full-paper reads limited; abstract-level for calibration papers; no multimodal calibration evidence exists. Open question to user: may LLM/VLM teachers be used at training time (not at inference)?

## Expansion trace
W1: 6 axes + skeptic -> leads coherence, token budgets, LoRA, curriculum. W2: coherence verified, LAVOIR/eve-rlcd/Laya code read, licences. User steering (non-LLM, 300k tok/s H200, <200ms): W3 throughput/serving/FLOPs; skeptic feasibility attack. Convergence: user-directed stop to build; remaining leads logged as gaps.

## Addendum (waves 4, after user steering 2-3: LLM teacher allowed, 300k tok/s dropped, target $0.030/M)
- Economics (DERIVED, unmeasured): Break-even 23-46k tok/s per GPU at 100% utilization; 100M-300M active params cost $0.001-0.007/M, 1B $0.01-0.022, 3B-active MoE not profitable. Ceiling ~1.1B active on H200 at 50% utilization for 50% margin; H200 no cheaper than H100; utilization and CPU media decode dominate [eval-repo, nar-heads, multimodal].
- Long context: 32k is a latency problem, not a cost one (dense ~0.2 s/request); hierarchical chunk->latent->global ~85 ms; accuracy on span-recall decisions unproven [nar-heads].
- Teachers: Apache panel (Qwen3, Qwen3-VL, Mistral-Small); Gemma outputs make students Model Derivatives; PoLL + cyclic permutation + explicit P(U) [pretrain, calibration].
- Independent evals of hosted decision models: 37 datasets 95-99% on easy sets, binary mis-thresholded at 0.5, hidden P(U), yes/no swap flips 50.5/100, trained small classifiers match decision models when labels exist, cascade matches accuracy at 0.43 cost [calibration, abstract-level].
- RL: Brier+bonus non-hackable for verbalized confidence (2607.04332); lead's toy check refutes 'r=c-p_a improper' for the policy-probability setting (debate round 3).
- Open: measured tok/s of 100M-1B encoders on a target GPU, KD vs continued-MLM ablation, chunk-latent accuracy, FP8 calibration.
