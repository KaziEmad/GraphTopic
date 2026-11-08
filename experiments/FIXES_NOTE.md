# Fixes to the full-dataset comparison (October 2026)

Files changed: `full_dataset_comparison_v1.py`, `baseline/baseline_models.py`.
None of the functions used for the cache hashes were changed, so finished
results and the Optuna studies are reused when the script is restarted.

## 1. Process killed on TopicNet/NIPS (system RAM)
- **Symptom:** `Killed` / `Command terminated by signal 9` right after
  "Final N topics generated"; peak RSS ~6.5 GiB on a shared 15 GiB machine.
- **Cause:** `calculate_coherence_scores` held every 2-3-gram of every
  long NIPS document in memory at once (unused `vectorizer.fit`, a full
  `tokens` list kept alive, parallel `CoherenceModel` workers).
- **Fix:** removed the unused `fit`, built the gensim `Dictionary` from a
  generator, `CoherenceModel(..., processes=1)`, freed each model after use.
- **Effect on results:** none intended. Verify once with
  `check_coherence_equivalence.py` (must print `IDENTICAL`).

## 2. TopMost device mismatch (ProdLDA, ECRTM, HyperMiner)
- **Symptom:** `Expected all tensors to be on the same device ... cuda:0 ...
  cpu`, reported as `status=no_topics`.
- **Cause:** data was put on the GPU (`RawDataset(device="cuda")`), but
  TopMost trainers never move the model; HyperMiner/SawETM also keep plain
  tensors (`curvature`, `gam_prior`, ...) and a `self.device` created on CPU.
- **Fix:** `_patch_topmost_trainers_device()` in
  `full_dataset_comparison_v1.py` moves the model, its plain tensors and its
  `device` attribute to the data's device just before `trainer.train()`.
  GPU assignment only; no change to the models or training.

## 3. FASTopic failed on every dataset
- **Symptom:** "unable to interpret topic as either a list of tokens or a
  list of ids" (gensim coherence) on all datasets; CUDA out of memory on the
  10k-doc chunks.
- **Cause:** `FASTopicTrainer` ignored the shared `NgramPreprocessing` and
  built its own single-word vocabulary (~44k words). Its topics were single
  words that the 2-3-gram coherence dictionary does not contain, and topic
  words were looked up in `dataset.vocab` (a different vocabulary). The large
  vocabulary also made FASTopic's full-batch training exceed GPU memory.
- **Fix (in `baseline_models.py`):** FASTopic now receives the same
  `NgramPreprocessing` (same `ngram_range`, same `max_features`) as the other
  TopMost models (`return_sparse=True`, because FASTopic's loader needs a
  sparse matrix), and its topic words are read from FASTopic's own vocabulary.
- **Effect on results:** FASTopic now uses the same n-gram vocabulary as the
  other neural baselines, so it is comparable with them. There were no
  earlier valid FASTopic results to compare against.

## 4. Failed results were cached and never retried
- **Fix:** chunk results with status `error`, `error_oom` or `no_topics` are
  re-run instead of loaded from the raw-results cache. For datasets that
  already have a final CSV, models whose rows show `error`, `error_oom`,
  `no_topics` or `all_failed` are re-run; all successful models load from
  cache.
- **Side effect:** a model that genuinely produces no topics is re-run on
  every restart (costs time, does not change results).

## Open item (not changed)
- BERTopic on one 10k-doc dataset had chunks with non-finite `c_npmi`; the
  averages silently use only the finite chunks (3 of 5). Worth checking and
  reporting consistently across models.
- Coherence `texts` are rebuilt from bag-of-words (de-duplicated, not in
  original order), which affects the sliding-window measures (c_v, c_uci,
  c_npmi). Unchanged for comparability; applies to all models equally.
