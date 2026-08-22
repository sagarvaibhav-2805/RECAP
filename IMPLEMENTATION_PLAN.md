# RECAP Implementation Plan (Simple Version)

This file explains, step by step, kaise RECAP paper ko actually build karna hai — kaunsa
data kahan se aayega, kya calculate karna hai, aur kaunsa file kis stage mein banega.
Har stage apna output CSV/JSON file save karta hai, taaki kaam resumable rahe aur
har (language, direction) combination independently chal sakein (jaise COMET scoring
mein tha).

---

## Scope (abhi ke liye)

Abhi sirf ye 6 directions cover karne hain — Hindi ↔ Tribal, jahan Tribal ∈
{Bhili, Gondi, Mundari}:

```
Hindi → Bhili,   Bhili → Hindi
Hindi → Gondi,   Gondi → Hindi
Hindi → Mundari, Mundari → Hindi
```

En2tgt / tgt2en (English wale directions) is scope mein nahi hain — filhaal skip.
Lekin poora pipeline **plug-and-play** tarah design karna hai (neeche section dekho),
taaki baad mein naya language ya naya direction (jaise English↔Tribal) add karna ho to
sirf ek config entry add karni pade, code kahin bhi rewrite na karna pade.

---

## Plug-and-Play Design (important — hamesha yaad rakhna hai)

Koi bhi stage (split, calibration, reward, pairs, DPO, GRPO, PPO, eval) kisi language,
direction, hyperparameter, ya ablation-setting ka value **hardcode** nahi karega.
Poora pipeline ek single Python file — **`config.py`** — se driven hoga. (Neeche
"Central Config" section mein iske sab andar ke parts detail mein hain.)

- Naya language/direction add karna ho (jaise `en2tgt` ya ek naya tribal language),
  to bas `config.py` ki list mein ek entry add karo — koi stage-script edit nahi
  karna.
- Har stage ka script `build_maha_data.py`/`build_maha_data_2.py` jaisa CLI pattern
  follow karega: `--lang <X> --direction <Y>` se ek combo chalao (single-GPU job ke
  liye), ya bina argument diye poori config-list pe loop chale (sequential mode) —
  jo already `build_maha_data.py` mein implement hai, wahi pattern sab stages mein
  reuse hoga.
- Folder paths (`recap_splits/<lang>/<direction>/...` etc.) automatically config se
  banenge, kabhi bhi `Bhili`, `Gondi`, `hi2tgt` jaise literal string kisi stage-logic
  (calibration formula, reward formula, DPO loss) ke andar nahi likhe jaayenge — sirf
  path-building mein use honge.
- Model list (`nllb, mt5, qwen, llama`) bhi isi tarah config-driven rahega, taaki
  future mein koi model add/remove karna ho to wahan bhi ek jagah change ho.
- Har hyperparameter (reward weights, margin threshold, DPO/GRPO/PPO settings) bhi
  `config.py` se aayega — koi script apne andar `beta = 0.1` jaisi value likhkar
  nahi rakhega.

Isse fayda: aaj sirf 6 Hindi↔Tribal directions chalenge, lekin kal agar English
directions ya koi naya tribal language add karna ho, to sirf `config.py` edit hoga —
har stage ka code same rahega. Aur koi bhi naya ablation/experiment chalane ke liye
bhi sirf `config.py` mein ek naya entry add hoga, kahin bhi code duplicate/edit nahi
karna padega.

---

## Central Config — `config.py`

Ye pura project ka **single source of truth** hai. Koi bhi stage-script kisi bhi
number, path, ya list ko apne andar hardcode nahi karega — sab `config.py` se import
hoga. Isse fayda: koi bhi naya experiment/ablation chalana ho to sirf yahan ek entry
add karni hai, kisi stage-script ko touch nahi karna.

`config.py` ke andar ye groups honge (Python dataclasses ya simple dicts ke roop
mein — JSON nahi, kyunki Python file mein defaults, validation, aur computed values
(jaise auto-built folder paths) sab ek jagah ho sakte hain):

1. **Data / plug-and-play list**
   - `LANGUAGES` (Bhili, Gondi, Mundari — future mein aur add ho sakte hain)
   - `DIRECTIONS` (hi2tgt, tgt2hi — future mein en2tgt/tgt2en add ho sakte hain)
   - `MODEL_NAMES` (nllb, mt5, qwen, llama)

2. **Paths**
   - Sab `recap_*` root folders (`recap_splits`, `recap_calib`, `recap_rewards`,
     `recap_pairs`, `recap_dpo`, `recap_grpo`, `recap_ppo`, `recap_eval`) — ek
     function jo `(lang, direction)` leke sahi path bana de, taaki koi bhi jagah
     path manually string-concat na kare.

3. **Split settings** (Stage 1)
   - Train/val/test ratio (80/10/10), random seed.

4. **Calibration settings** (Stage 2)
   - Kaunse 5 quantities standardize honge, epsilon value (division-by-zero se
     bachne ke liye).

5. **Reward settings** (Stage 3) — ye sabse important hai ablations ke liye. Paper
   mein DO alag ablation-studies hain jo dikhne mein similar hain lekin different
   cheez test karte hain — dono ko alag naming se rakhna hai taaki confuse na ho:

   **(a) Preference-construction cumulative ablation** (paper §8.3, Table 8 —
   "kya har penalty/filter alag se help karta hai" test karta hai):
   - `ablation_quality_only` → sirf BLEU/chrF++/COMET, penalties = 0, koi margin
     filter nahi
   - `ablation_quality_plus_rep` → + repetition penalty
   - `ablation_quality_plus_len` → + length penalty
   - `ablation_full_reward` → dono penalties, validation-selected weights
   - `ablation_full_reward_margin` → + margin filtering (δ threshold)
   - `ablation_full_recap` → + balanced sampling (ye `recap_dpo` jaisa hi hai)

   **(b) Main experiment matrix** (paper §11.9 — "final headline result" ke liye,
   Table 7 ka source; ye alag-alag safety-mechanism ko ek-ek karke off karta hai,
   baaki sab RECAP jaisa hi rakhta hai):
   - `dpo_raw` → uncalibrated raw metric average se pairs (koi z-score calibration
     nahi, koi penalty nahi, koi margin filter nahi, koi balancing nahi) — sabse
     simple baseline.
   - `dpo_quality_only` → calibrated BLEU/chrF++/COMET (z-score wala), penalties=0,
     **lekin dedup + margin filter + balanced sampling ON rehte hain** (ye #(a) ke
     `ablation_quality_only` se DIFFERENT hai — wahan margin filter OFF tha).
   - `dpo_no_confidence` → full calibrated reward (penalties sahit) + dedup +
     balancing, lekin δ=0 (margin filtering OFF).
   - `recap_dpo` → full pipeline (sab kuch ON) — RECAP-DPO ka primary result.

   Naya ablation chalana ho to bas ek naya preset-entry add karo — kahin bhi code
   nahi likhna.

6. **Pair-mining settings** (Stage 4 & 5)
   - Confidence margin `δ`.
   - Balanced-sampling cap `q_d` per generator-pair-type.

7. **DPO settings** (Stage 6)
   - `beta`, learning rate, batch size, epochs, max sequence length,
     confidence-weighting on/off.

8. **GRPO settings** (Stage 7 — scratch implementation)
   - Group size `G` (2 ya 3), sampling temperature/nucleus/max-length/repetition
     control, learning rate, update frequency.
   - **Fixed source-subset size** — GRPO poore train-split pe nahi chalta (fresh
     rollout generate karna mehenga hai), ek fixed subset use hota hai: default
     50K sources per direction (jitna available ho, kam se kam utna), data-size
     ablation ke liye `[10K, 25K, 50K]` list bhi.

9. **PPO settings** (Stage 8 — TRL-based)
   - Clip epsilon, KL coefficient `β_KL`, value-loss weight `c_V`, GAE `γ,λ`,
     rollout batch size, epochs.

10. **Experiment registry**
    - Ek jagah jahan har **experiment/ablation ka poora combo** define hota hai:
      kaunsa reward-preset (#5 se), kaunsa trainer-config (#7/#8/#9 se), aur
      output-folder ka naam (jaise `recap_dpo/<lang>/<direction>/quality_only/`,
      `recap_dpo/<lang>/<direction>/full_recap/` — alag ablations ke checkpoints
      kabhi ek dusre ko overwrite nahi karenge).
    - Stage-scripts sirf `--experiment <name>` CLI argument lenge (plus already
      established `--lang`/`--direction`), aur baaki sab `config.py` ki us entry se
      khud-ba-khud aa jaayega.
    - **Main matrix ke 8 conditions** (paper §11.9, minimum core matrix — Table 7
      isi se banega) yahan named entries ke roop mein honge:
      1. `sft` — untouched baseline, koi training nahi
      2. `dpo_raw`
      3. `dpo_quality_only`
      4. `dpo_no_confidence`
      5. `recap_dpo` — primary result
      6. `sft_ppo` — PPO seedha SFT se
      7. `sft_grpo` — "pure GRPO" ablation, SFT se seedha (jab compute allow kare)
      8. `recap_dpo_grpo` — RECAP-DPO+GRPO (GRPO refinement `recap_dpo` checkpoint
         ke upar) — ye asli headline result hai
    - **8 targeted ablations** (paper §11.9, RECAP-DPO ke baad, appendix tables ke
      liye) bhi named presets honge: candidate-diversity (heterogeneous vs
      mT5-self-sampling), calibration-method, reward-components (#(a) se reuse),
      confidence/δ-grid, pair-selection-strategy (all-pairs vs best-worst vs
      balanced), data-scale (10K/25K/50K), GRPO group-size (`G=2` vs `G=3`), aur
      weight-sensitivity (selected weights vs neighboring simplex-grid values).

11. **Reproducibility settings**
    - `SEED` — ek default seed.
    - `SEEDS` — multiple-seed list (paper ke liye mean±std report karna hai — Section
      8.1), e.g. `[13, 42, 2026]`.

12. **DDP / distributed settings**
    - Backend (`nccl`), distributed timeout value, env-var flags
      (`NCCL_ASYNC_ERROR_HANDLING=1` waghera).

13. **Inference config**
    - Decoding settings (greedy/beam, beam size, max length) jo `recap_infer.py`
      aur Stage 9 ka eval dono use karenge — ek hi jagah se.

14. **Best-checkpoint registry**
    - Stage 9 evaluation ke baad, har `(lang, direction)` ke liye jo checkpoint
      jeeta (SFT/DPO/DPO+GRPO/PPO) uska path yahan fill hoga — `recap_infer.py`
      isi se resolve karega ki kaunsa checkpoint load karna hai.

Isse pura pipeline reproducible aur comparable ban jaata hai: koi bhi do experiments
ka result compare karna ho, to sirf unke `config.py` entries dekh ke pata chal
jaayega ki kya-kya alag tha.

---

## Reproducibility (paper ke liye zaroori)

Paper mein jo bhi number report honge, wo baad mein dobara-produce-karne-layak
(reproducible) hone chahiye. Isliye:

- Ek shared `set_seed(seed)` function (in `recap_utils.py`) — Python `random`,
  `numpy`, `torch` (CPU + CUDA), aur `transformers.set_seed()` sab ek saath seed
  karega. DDP mein har rank ka data-shuffle seed `seed + rank` se deterministically
  nikalega, taaki same job dobara chalane pe same data-order mile.
- `torch.backends.cudnn.deterministic = True`, `benchmark = False` — GPU pe 100%
  bitwise determinism kabhi kabhi possible nahi hoti (kuch CUDA ops non-deterministic
  hote hain); jahan aisa ho, use honestly document karenge, chhupayenge nahi. Isi
  liye paper "same statistics across reruns" target karega, na ki "bit-for-bit
  identical output".
- **Har training run apna `run_manifest.json` save karega** (checkpoint folder ke
  andar): git commit hash, us run ka resolved `config.py` snapshot, library
  versions (`torch`, `transformers`, `trl`, `accelerate`), GPU count/type, aur
  seed. Isse paper ka har reported number exactly retrace ho sakta hai.
- **Multiple seeds**: `config.py` ki `SEEDS` list ke har seed ke liye alag output
  subfolder banega (`recap_dpo/<lang>/<direction>/<experiment>/seed_<n>/`) —
  training/eval seed ke hisaab se loop karega, jaisa `--lang`/`--direction`/
  `--experiment` ke liye already karte hain.
- **Final reported translations ke liye greedy decoding** (koi randomness nahi) —
  sirf GRPO ki training-time sampling stochastic rahegi (wahi RL ka poora point
  hai), uska seed bhi manifest mein log hoga.
- Stage 1 ka split-shuffle aur Stage 5 ka balanced-sampling — dono ka seed bhi
  `config.py` se aayega, randomly generate nahi hoga.

---

## Deadlock-free DDP (multi-GPU training)

DDP mein deadlock ka matlab: ek process kisi collective operation (jaise gradient
sync) ka wait kar raha hai jo kabhi aayega hi nahi, kyunki koi doosra process
wahan tak pahunch hi nahi paya. Ye rules follow karenge taaki aisa kabhi na ho:

- **Raw `torch.distributed` khud se nahi likhenge** — HF `accelerate` library
  (`Accelerator()`) use karenge. TRL ke `DPOTrainer`/`PPOTrainer` already isi pe
  built hain — unke liye DDP matlab bas `accelerate launch --num_processes=<N>`
  se chalana hai.
- **GRPO (scratch) mein bhi `Accelerator()` hi use karenge**, raw DDP nahi.
- **Frozen reference models (π_ref) kabhi DDP-wrap nahi honge** — sirf trainable
  policy DDP mein jaayega. Frozen model har process apni local copy rakhega
  (`no_grad()`, `.eval()`).
- **GRPO ka advantage computation (μ_R, σ_R) hamesha per-source, per-process local
  rahega** — koi custom cross-process collective GRPO mein nahi likhenge. Isse sirf
  DDP ka apna automatic gradient-sync collective chalta hai, jo already reliable
  hai.
- **Uneven batches se bachne ke liye** `DistributedSampler(..., drop_last=True)` —
  taaki har process ko exact same number of batches milein.
- **Env vars set karenge taaki hang crash mein convert ho jaaye** (silent-forever-
  hang HPC pe wall-time barbaad karta hai): `NCCL_ASYNC_ERROR_HANDLING=1`, aur
  `init_process_group(timeout=...)` ek reasonable value pe.
- **Rank-conditional code sirf I/O (logging, checkpoint-save) ke liye** — forward/
  backward/optimizer-step kabhi kisi rank-condition ke andar nahi.
- **Checkpoint save se pehle/baad `accelerator.wait_for_everyone()`** — taaki fast
  processes slow process ke disk-write se pehle aage na badh jaayen.
- **Mid-training validation (COMET) sirf main process pe chalega** — humne pehle
  hi COMET + multi-GPU se OOM bug jhela hai (Lightning ka internal GPU
  auto-select). Training ke beech validation-COMET sirf ek process pe chalega,
  baaki `wait_for_everyone()` pe barrier karenge.
- Ye sab helpers (`set_seed`, DDP-safety wrappers, manifest-saving) `recap_utils.py`
  mein rahenge, taaki teeno training scripts (DPO/GRPO/PPO) mein ye code duplicate
  na ho.

---

## Folder Structure

```
RECAP/
  config.py                                 # single source of truth (Central Config)
  recap_utils.py                            # shared: set_seed, DDP-safety helpers, run_manifest
  recap_reward.py                           # shared: RewardEngine class (fit/score_batch/serialize)
  recap_preference.py                       # shared: PreferenceBuilder class (Stage 4 ke liye)
  recap_checks.py                           # shared: 9 pre-flight automated checks (Stage 0)
  recap_infer.py                            # single inference entrypoint + translate() function
  maha_data_2/<lang>/<direction>.csv        # already ready — source, gold_truth, per-model prediction/BLEU/chrF++/COMET
  recap_splits/<lang>/<direction>/{manifest.csv, train,val,test}.csv
  recap_calib/<lang>/<direction>/stats.json     # RewardEngine.serialize() output
  recap_rewards/<lang>/<direction>/rewards.csv
  recap_pairs/<lang>/<direction>/{pairs_all.csv, pairs_balanced.csv}
  recap_dpo/<lang>/<direction>/<experiment>/seed_<n>/{checkpoint/, run_manifest.json}
  recap_grpo/<lang>/<direction>/<experiment>/seed_<n>/{checkpoint/, run_manifest.json}
  recap_ppo/<lang>/<direction>/<experiment>/seed_<n>/{checkpoint/, run_manifest.json}
  recap_eval/<lang>/<direction>/<experiment>/report.json
  recap_report/tables/{table3.csv, ..., table9.csv}
  recap_report/plots/{figure1.png, figure2.png, figure3.png}
  recap_human_eval/sample_500.csv               # Table 10 — annotate karne ke liye
  recap_human_eval/labels_500.csv               # Table 10 — annotation aane ke baad
```

`<lang>` = Bhili / Gondi / Mundari, `<direction>` = hi2tgt / tgt2hi. Har jagah ye 6
combinations independent rahenge — ek ka calculation dusre ko touch nahi karega.

---

## Code Files (Overview)

Sab files ek jagah, taaki pata chale kaunsi file kya karegi:

| File | Kaam |
|---|---|
| `config.py` | Sab settings/hyperparameters/experiment-registry — single source of truth |
| `recap_utils.py` | `set_seed()`, DDP-safety helpers, `save_run_manifest()` — teeno trainers isse import karenge |
| `recap_reward.py` | `RewardEngine` class (`fit()`, `score_batch()`, `serialize()`) — Stage 2/3, DPO pair-mining, PPO, GRPO sab isi ek class ko reuse karenge |
| `recap_preference.py` | `PreferenceBuilder` class — Stage 4 (pairs banane ka actual logic) |
| `recap_checks.py` | Stage 0 — 9 pre-flight automated checks, expensive run se pehle chalega |
| `recap_split.py` | Stage 1 — train/val/test split + manifest/audit |
| `recap_calibrate.py` | Stage 2 — `RewardEngine.fit()` chalata hai, `stats.json` (`serialize()` output) banata hai |
| `recap_score.py` | Stage 3 — `RewardEngine.score_batch()` chalata hai, `rewards.csv` |
| `recap_mine_pairs.py` | Stage 4 — `PreferenceBuilder` use karke pairwise margin filtering, `pairs_all.csv` |
| `recap_balance_pairs.py` | Stage 5 — balanced sampling, `pairs_balanced.csv` |
| `recap_train_dpo.py` | Stage 6 — TRL `DPOTrainer` |
| `recap_train_grpo.py` | Stage 7 — scratch GRPO loop |
| `recap_train_ppo.py` | Stage 8 — TRL `PPOTrainer` |
| `recap_infer.py` | Stage 10 — ek hi inference entrypoint (`translate()` function) |
| `recap_evaluate.py` | Stage 9 — test-set report; translation ke liye `recap_infer.py` ka `translate()` reuse karega |
| `recap_report_tables.py` | Stage 11 — Tables 3-9 saved outputs se generate karta hai |
| `recap_report_plots.py` | Stage 11 — Figures 1-3 generate karta hai |
| `recap_sample_for_human_eval.py` | Stage 11 — Table 10 ke liye 500-pair sample nikalta hai |
| `recap_human_eval_analysis.py` | Stage 11 — human labels aane ke baad Table 10 compute karta hai |

Sab stage-scripts (split se lekar eval tak) `--lang --direction --experiment`
(aur jahan applicable `--seed`) CLI arguments lenge — same pattern jo
`build_maha_data.py` mein hai.

---

## Stage 0 — Pre-flight Automated Checks (`recap_checks.py`)

Expensive training run se PEHLE, ye 9 checks automatically chalne chahiye (paper
§11.12) — inme se koi bhi fail ho to run start hi nahi hoga:

1. Reward weights non-negative hain aur sum-to-1 (numerical tolerance ke andar).
2. Quality fix rakh ke, repetition ya length mismatch badhane se reward kam hi
   hona chahiye, kabhi badhna nahi chahiye (monotonicity check).
3. Sab standardized quantities sirf saved training-only stats use kar rahi hain,
   aur sab finite hain (NaN/Inf nahi).
4. Identical completions kabhi ek DPO pair nahi bante, aur har retained pair ka
   reward-margin positive hai.
5. Koi bhi source-identifier ek se zyada split mein nahi hai, aur koi validation/
   test row pair-mining mein nahi gaya.
6. DPO/PPO/GRPO ke reference-policy parameters training ke dauran unchanged rahe
   (koi gradient nahi mila).
7. GRPO/PPO ke completions **current policy se live generate** hue the, static
   `maha_data_2` candidate files se load nahi hue.
8. Har rollout reward ko sahi source, reference, candidate, aur direction mila.
9. Har final checkpoint, manifest, pair-file, calibration-file, metric-config,
   seed, aur trainer-state ek hash ke saath record hua hai.

Agar kisi direction mein koi issue mile (checkpoint access missing, valid pairs
kam, COMET coverage na ho, metric fail, RL unstable), to us direction ka apna
alag failure-report likho (counts, distributions, logs, attempted safeguards) —
kisi doosri direction ka data/result kabhi substitute mat karo.

---

## Stage 1 — Data ko Train/Val/Test mein baanto

- Input: `maha_data_2/<lang>/<direction>.csv`.
- **Pehle ek source-level manifest banao** — har source/reference row ko ek stable
  identifier do, aur source-checksum + reference-checksum bhi store karo. Ye
  manifest pair-construction se PEHLE banta hai.
- **Data validation — reject-and-log, silent-drop nahi**: missing source/reference
  text, invalid candidate-column mapping, empty/malformed candidates, ya
  non-finite metrics waale rows ko reject karo aur ek log file mein likho kyun
  reject hua — chupke se drop mat karo.
- **Duplicate-aware split**: exact duplicates, normalized duplicates, aur near
  duplicates waale sources **same split mein hi rehne chahiye** — warna train aur
  test ke beech leakage ho sakta hai (ek duplicate train mein, uska near-duplicate
  test mein — ye galat "achha result" dikhayega).
- Rows (sources) ko ek fixed random seed se shuffle karo, phir 80% train, 10% val,
  10% test mein baanto.
- Ye split har (lang, direction) ke liye ALAG se karna hai — Bhili ka split Gondi se
  share nahi hoga.
- Rule: ek source sirf ek hi split mein aana chahiye (train ya val ya test, kabhi do
  jagah nahi).
- Output: `recap_splits/<lang>/<direction>/manifest.csv` (source-id, checksums,
  direction, split, candidate-column-mapping, split-seed) + `{train,val,test}.csv`.

---

## Stage 2 — Calibration (`RewardEngine.fit()`)

Idea: ek BLEU score of 30 Hindi→Bhili ke liye "normal" ho sakta hai, lekin
Bhili→Hindi ke liye "kam" ho sakta hai. Isliye har direction ke apne average aur
spread (mean/std) nikaalne padte hain, taaki scores compare-karne-layak ban sakein.

Ye stage `recap_reward.py` ke `RewardEngine` class ka `fit(training_candidates)`
method call karta hai (§11.3):

- **μ aur σ har (language, direction) combo ke liye ALAG-ALAG calculate honge** —
  yani Bhili/hi2tgt, Bhili/tgt2hi, Gondi/hi2tgt, Gondi/tgt2hi, Mundari/hi2tgt,
  Mundari/tgt2hi — in 6 combos ka apna-apna `RewardEngine` banega. Kisi bhi combo ke
  stats kabhi doosre combo ke saath mix/share nahi honge, kyunki har direction ka
  score-distribution alag hota hai (Hindi→Bhili aur Bhili→Hindi ke normal scores
  bhi alag ho sakte hain).
- Sirf **train** split se calculate karna hai (val/test se kabhi nahi).
- In 5 quantities ka mean (μ) aur std-dev (σ) nikaalo (sab 4 models ke train rows
  milaake, sirf usi ek combo ke andar):
  1. BLEU
  2. chrF++
  3. COMET
  4. Repetition score (`P_rep`) — output mein kitna text repeat ho raha hai (repeated
     word n-grams ka ratio).
  5. Length mismatch score (`P_len`) — candidate ki length reference se kitni alag hai.
- `fit()` ye bhi validate karta hai ki stored BLEU/chrF++ verifiably comparable hain
  (same tokenization/smoothing config se aaye) — agar nahi, to candidate+reference
  se dobara compute karega ek fixed documented config se.
- `RewardEngine.serialize()` ye save karta hai (`stats.json`): μ/σ har quantity ke
  liye, checkpoint identifiers, tokenization/smoothing configuration, calibration
  values, reward-weight hyperparameters, aur ek **configuration hash** — taaki
  baad mein exactly wahi RewardEngine reload ho sake, kabhi drift na ho.
- Ye file baaki sab stages ke liye zaroori hai — kabhi bhi val/test data se recompute
  mat karna, na hi kisi rollout-group pe refit karna (DPO validation, PPO/GRPO
  rollouts, final test-analysis — sab jagah frozen `stats.json` hi use hoga).

---

## Stage 3 — Har translation ko ek "Reward" score do (`RewardEngine.score_batch()`)

Ye stage `RewardEngine.score_batch(sources, candidates, references)` call karta
hai — same method GRPO/PPO rollout-scoring aur eval-time bhi reuse karenge, taaki
reward-computation kahin duplicate na ho.

Har row ke har model (NLLB/mT5/Qwen/Llama) ke prediction ke liye:

1. **Pehle validity check** — empty output, extreme length, non-finite scores, ya
   unsupported/malformed characters waale completions ko ek explicit
   `invalid`/exclusion flag do. Ye flagged completions kabhi accidentally high
   reward na paayein — inhe reward-computation se pehle hi flag karna hai, baad
   mein filter karna nahi bhoolna.
2. Stored BLEU/chrF++ (jo 0-100 scale pe hain) ko 0-1 scale pe le aao (divide by 100).
3. Stage 2 ke frozen μ/σ (saved `stats.json` se) use karke BLEU, chrF++, COMET,
   P_rep, P_len — sabko "standardize" karo (formula: `(value - mean) / (std +
   chhota_number)`).
4. Quality score = teeno (BLEU + chrF++ + COMET) ka average.
5. Final Reward = `w_quality × Quality - w_rep × Repetition - w_len × LengthMismatch`
   — teeno weights (w_quality, w_rep, w_len) add hoke 1 hone chahiye, aur validation
   data se choose karne hain (paper ka example: 0.60, 0.25, 0.15).
6. Ye poora vector (BLEU, chrF++, COMET, Rep, Len, Reward, validity-flag) save karo
   har row × har model ke liye.
- Output: `recap_rewards/<lang>/<direction>/rewards.csv` — train, val, test sab rows
  ke liye (val/test sirf reporting ke liye use hoga, training pairs banane ke liye
  nahi).

---

## Stage 4 — "Kaunsa translation better hai" wale pairs banao (`PreferenceBuilder`)

Ye stage `recap_preference.py` ke `PreferenceBuilder` class se banega — sirf
training sources, stored candidates, aur fitted `RewardEngine` use karke.

Sirf **train** data pe:

- Har source sentence ke 4 models ke output hain → in 4 mein se har 2-2 ka pair
  banao (total 6 pairs per source: NLLB-mT5, NLLB-Qwen, NLLB-Llama, mT5-Qwen,
  mT5-Llama, Qwen-Llama).
- **Invalid-flag waale candidates** (Stage 3 se) kisi pair mein use nahi honge.
- Har pair ke liye dono ke reward ka farak (`Δ`) nikaalo.
- Agar `|Δ|` ek threshold (`δ`, "confidence margin") se zyada hai, tabhi is pair ko
  rakho — warna discard, kyunki farak bahut chhota hai to pata nahi kaun sach mein
  better hai.
- Jo zyada reward wala hai use "better" (y+) maano, doosre ko "worse" (y-).
- Duplicate pairs hatao (agar dono outputs same text hain).
- Har pair record mein rakho: source, gold-reference, chosen+rejected completion,
  source-identifier, generator labels, raw metric vectors, standardized
  components, final rewards, reward-margin, filter-decision, split, random seed,
  aur configuration hash — taaki koi bhi pair baad mein fully auditable ho.
- Output: `recap_pairs/<lang>/<direction>/pairs_all.csv` — ye file **immutable**
  hai; DPO isse read karta hai, kabhi live-rescore nahi karta.

---

## Stage 5 — Pairs ko balance karo

Problem: agar sirf ek jodi (jaise mT5-vs-Qwen) hi baar baar aa rahi hai, to training
biased ho jaayegi. Isliye:

- Har pair-type (upar wale 6 types) mein se ek maximum limit (`cap`, validation se
  decide karna) tak hi pairs rakho.
- **Reward-margin bins ke across bhi balance karo** (jahan data allow kare) — sirf
  generator-pair-type balance karna kaafi nahi, warna sab retained pairs high-margin
  (aasan) ya sab low-margin (mushkil) ho sakte hain — dono tarah ke examples chahiye.
- **Per-source cap bhi rakho** — koi ek source apne 6 possible pairs se zyada
  contribute nahi karega, taaki chand sources training ko dominate na karein.
- Kisi bhi pair ki "kaun better hai" wali direction ko flip mat karo — sirf count
  control karo.
- Check karo har model kitni baar "preferred" aur kitni baar "rejected" bana — agar
  koi ek model hamesha jeet/haar raha hai to wo warning sign hai.
- Output: `recap_pairs/<lang>/<direction>/pairs_balanced.csv` — yahi final training
  data hai DPO ke liye.

---

## Stage 6 — DPO Training (main training step)

- Model: mT5 (kyunki validation pe sabse strong average model yahi nikla).
- Trainable model (`π_θ`) start hota hai existing mT5 SFT checkpoint se.
- Ek "reference model" (`π_ref`) bhi rakho — same checkpoint ki ek copy, jo kabhi
  update nahi hogi (fix rahegi).
- Har pair (y+, y-) ke liye 4 cheezein chahiye:
  1. Trainable model, y+ ko kitni probability deta hai
  2. Trainable model, y- ko kitni probability deta hai
  3. Reference model, y+ ko kitni probability deta hai
  4. Reference model, y- ko kitni probability deta hai
- Loss formula training ko is taraf push karta hai: trainable model apni SFT
  baseline se zyada y+ ko prefer kare, aur y- ko kam.
- Training ke dauran monitor karo: loss, average margin, reference se kitna
  door gaya (KL), validation BLEU/chrF++/COMET, aur kitna output "degenerate"
  (bekaar/repeat) ban raha hai.
- Hyperparameters (beta, learning rate, batch size, epochs) validation data se
  chuno — best validation checkpoint rakho, last step wala nahi.

### TRL se kaise implement karenge

- Hugging Face ke `trl` library ka `DPOTrainer` use karenge — ye seq2seq (encoder-decoder)
  models ko achhe se support karta hai, to mT5 ke saath directly kaam karega.
- Dataset format jo `DPOTrainer` chahta hai: columns `prompt`, `chosen`, `rejected` —
  hamari `pairs_balanced.csv` (`source, y+, y-`) ko bas rename karke isi format mein
  convert karna hai, extra kuch nahi.
- Model: `AutoModelForSeq2SeqLM.from_pretrained(<mT5 SFT checkpoint>)`; reference
  model bhi wahi checkpoint se doosri copy load karo (frozen).
- `DPOConfig` mein paper ke symbols map honge: `beta` = paper ka β, `learning_rate`,
  `per_device_train_batch_size`, `num_train_epochs`, `max_length`/`max_prompt_length`.
- Confidence-weighting (per-pair `c_i` weight) TRL ke default loss mein nahi hai —
  agar use karna ho to ek chhota trainer subclass/override likhna padega.
- `trl` ka exact version pin karke rakhna hai `requirements.txt` mein, kyunki iska
  API version-to-version badla hai.

---

## Stage 7 — GRPO (DPO ke baad optional refinement)

GRPO ke DO alag conditions hain (dono `config.py` ke Experiment Registry mein
separate entries — Stage 6 ke pattern se):

- **`sft_grpo`** ("pure GRPO" ablation) — policy aur frozen KL-reference dono
  **seedha original mT5 SFT checkpoint** se init honge (DPO bilkul involve nahi).
- **`recap_dpo_grpo`** (primary refinement, headline result) — policy `recap_dpo`
  ke trained checkpoint se init hoga, frozen KL-reference bhi usi DPO checkpoint
  ki ek frozen copy hogi (SFT nahi).

Dono conditions ke liye training loop same hai:

- Har (fixed-subset wale) source ke liye current model se G∈{2,3} fresh
  translations sample karo (group) — poore train-split pe nahi, sirf fixed
  source-subset pe (Central Config #8 dekho).
- Har ek ko Stage 3 wala reward formula se score karo — **usi frozen calibration
  stats (`stats.json`) se jo Stage 2 mein train-split se fit hui thi** — rollout
  ke waqt calibration dobara-fit nahi hogi.
- Group ke andar advantage nikaalo: `(reward - group_mean) / sqrt(group_variance + ε)`.
  Agar group variance effectively zero hai (sab samples ka reward same aaya), to
  us group ka advantage 0 set karo ya group ko skip karo — ek fixed, documented
  rule follow karna hai, kabhi random decide nahi.
- **Update rule confirm ho gaya (paper page 37)**: GRPO bhi PPO jaisa **clipped
  objective + KL regularization (frozen reference ke against)** use karta hai —
  sirf value-head PPO se nahi hota (GRPO ka baseline group-mean se aata hai, alag
  value-network ki zaroorat nahi).
- Training ke dauran BLEU/chrF++/COMET/repetition/length sab monitor karo — agar
  reward badh raha hai lekin repetition/length kharab ho rahi hai, to model "reward
  hack" kar raha hai (galat tareeke se score badha raha hai).
- Verify karna hai (test run mein): groups kabhi alag sources mix nahi karte, har
  completion individually score hoti hai, aur static `maha_data_2` candidate files
  kabhi rollout action ke roop mein reuse nahi hote (GRPO hamesha FRESH generation
  use karta hai, stored NLLB/mT5/Qwen/Llama outputs nahi).

### Implementation: scratch se, TRL use nahi karenge

- TRL ka `GRPOTrainer` primarily causal LM (decoder-only, jaise Qwen/Llama) ke liye
  design hua hai — mT5 jaisa encoder-decoder model iske saath reliably kaam karega
  ya nahi, ye pakka nahi hai (na hi easily verify ho sakta hai bina TRL internals
  fight kiye).
- Decision: GRPO **khud se (scratch se) likhenge**, TRL pe depend nahi karenge. Iski
  wajah — GRPO ka math already simple hai (group sample → Stage-3 wale reward se
  score → group ke andar advantage nikaalo → us advantage se weighted policy-gradient
  update) — apna chhota loop likhna, TRL ke unsupported-architecture internals se
  fight karne se zyada reliable hoga.
- Reward computation ke liye wahi shared `RewardEngine.score_batch()` reuse hoga
  jo Stage 2/3 mein fit/use hua — ek hi jagah reward-formula, kahin bhi duplicate
  nahi.
- Exact update rule (clipping chahiye ya nahi, KL penalty kaise) likhte waqt decide
  karenge — paper ke Eq. 55-58 se jo diya hai (μ_R, σ_R, advantage A_i) wahi base
  rahega.

---

## Stage 8 — PPO (comparison ke liye ek alag baseline)

- Ye preference pairs use nahi karta — seedha source sentence se translation
  generate karke reward deta hai.
- Model bhi SFT checkpoint se start hota hai, reference bhi frozen SFT copy.
- Har generated translation ko poora complete hone ke baad ek score milta hai
  (Stage 3 wala reward formula).
- Ek "value head" (extra chhota model) bhi training ko stable karne ke liye
  chahiye hota hai.
- Ye sabse zyada complex part hai — isliye recommend hai ki khud se pura PPO
  likhne ki jagah, ek existing library (TRL ka `PPOTrainer`) use karein, aur
  sirf apna reward function (Stage 3) usmein plug karein.

### TRL se kaise implement karenge

- Model wrapper: `AutoModelForSeq2SeqLMWithValueHead` (TRL ka seq2seq value-head
  wrapper) — plain `AutoModelForSeq2SeqLM` PPO ke liye nahi chalega, value head
  zaroori hai.
- Dataset: sirf `recap_splits/<lang>/<direction>/train.csv` se `source` column
  chahiye — PPO khud translations generate karta hai, koi pre-built pair nahi
  chahiye (DPO ke ulat).
- Reward function: TRL default ek learned reward *model* expect karta hai, lekin
  custom scalar reward bhi de sakte hain — rollout ke baad manually reward compute
  karke `ppo_trainer.step()` ko pass karo. Yahi wahi shared `RewardEngine.score_batch()`
  hai jo Stage 2/3 mein use hua (DPO pair-mining, PPO, GRPO — teeno isi ek
  method ko reuse karenge, taaki reward-formula kabhi mismatch na ho).
- Config mapping: β_KL (KL penalty) → PPOConfig ka `init_kl_coef`/KL setting,
  clip ε → `cliprange`, value-loss weight `c_V` → `vf_coef`, GAE ke γ,λ → `gamma`,
  `lam`.
- Caveat: TRL ke `PPOTrainer` ka API version-to-version kaafi badla hai — jo bhi
  `trl` version pin karo, uske exact docs se arg-names match karna zaroori hai
  code likhne se pehle.

---

## Stage 9 — Final Comparison aur Testing (`recap_evaluate.py`)

Test data (jo kabhi training/validation mein use nahi hua) pe, `recap_infer.py`
ka `translate()` function reuse karke, har condition ke liye decode karo — model
sirf source dekhta hai, kabhi reference nahi (ye rule SFT eval, DPO validation,
PPO/GRPO rollouts, aur final test decoding — sab jagah lagu hoga).

**Kya-kya report karna hai, per direction:**

- **Teen quality metrics alag-alag** — BLEU, chrF++, COMET — kabhi ek single
  number mein collapse nahi karna (paper explicitly mana karta hai).
- **Do degeneration diagnostics** — ye training-time wale standardized penalties
  (`P_rep`, `P_len` — Stage 2/3) se ALAG hain, reporting ke liye simple raw ratios
  hain:
  - `ρ_len = |y| / |y*|` (candidate/reference length ka seedha ratio)
  - `ρ_rep = repeated_ngrams / total_ngrams`
- **Delta from SFT baseline** — `ΔBLEU = BLEU(condition) - BLEU(SFT)`, waise hi
  `ΔChrF++`, `ΔCOMET` — har condition ke liye.
- Corpus-level (poore test-set pe ek BLEU/chrF++/COMET) AND sentence-level-average
  dono report karna hai, alag-alag.
- **Statistical significance**: SFT se key changes ke liye paired bootstrap
  confidence intervals (ya koi aur documented paired significance test) — sirf
  point-estimate kaafi nahi hai paper ke liye.
- Six directions ka result pehle individually dikhao, macro-average (unweighted,
  sab directions equal-weight) baad mein — kisi bhi bade direction ko conclusion
  dominate nahi karne dena.
- Agar metrics disagree karte hain (jaise BLEU up but COMET down), to trade-off
  report karo — sirf favorable metric mat select karo.

**Kaunse experiments compare honge** (`config.py` Experiment Registry se):

1. **Main matrix** (§Central-Config #10(b)) — `sft, dpo_raw, dpo_quality_only,
   dpo_no_confidence, recap_dpo, sft_ppo, sft_grpo, recap_dpo_grpo` — ye Table 7
   (headline result) banata hai.
2. **Preference-construction cumulative ablation** (§Central-Config #10(a)) —
   `ablation_quality_only → ... → ablation_full_recap` — ye Table 8 banata hai.
3. **8 targeted ablations** (candidate-diversity, calibration, reward-components,
   confidence, pair-selection, data-scale, GRPO group-size, weight-sensitivity) —
   Table 9 aur appendix tables banate hain.

Har ek ke liye `--experiment <name>` pass karke `recap_evaluate.py` chalega —
koi bhi naya comparison chahiye ho to bas `config.py` mein naya preset add karo.

---

## Stage 10 — Inference Script (`recap_infer.py`)

Ek hi standalone file, jo source sentence leke translation deta hai — training se
alag, apna khud ka file.

- Input: `--lang`, `--direction`, `--source` (ek sentence ya ek CSV of sources).
- Kaunsa checkpoint use karega, ye hardcode nahi hoga — `config.py` ki
  `BEST_CHECKPOINT` registry se resolve hoga (Stage 9 ke baad, jo direction jeeta
  wahi yahan fill hota hai) — chaaho to `--checkpoint`/`--experiment` se manually
  override bhi kar sakte ho.
- Decoding settings (greedy/beam, max length) `config.py` ke `INFERENCE_CONFIG`
  se aayenge — hardcoded nahi, taaki reporting-time behavior kabhi silently na
  badle.
- **Iske andar ek `translate()` function hoga jo `recap_evaluate.py` (Stage 9)
  bhi import karke reuse karega** — isse "jo hum evaluate kar rahe hain" aur "jo
  hum deliver kar rahe hain" kabhi drift nahi karenge, dono ek hi mechanism use
  karenge.
- CLI se bhi chalega (`python recap_infer.py --lang Bhili --direction hi2tgt
  --source "..."`) aur Python function ke roop mein bhi import ho sakega.

---

## Stage 11 — Reporting: Paper ke sab Required Tables/Figures

Paper explicitly 11 tables aur 3 figures maangta hai. Har ek yahan map kiya hai ki
kaunsa script/data usse banayega — taaki koi bhi required artifact bina-generator
ke na reh jaaye:

| Paper Item | Kya hai | Kaise banega |
|---|---|---|
| Table 1 | Related-work comparison | Manual/static — code se nahi banega, likhna hai |
| Table 2 | CSV schema | Manual/static — already documented hai |
| Table 3-5 | Per-language reference-model BLEU/chrF++ (val+test) | `recap_report_tables.py` — `maha_data_2`/`recap_rewards` ke stored BLEU/chrF++ se direct |
| Table 6 | mT5-backbone-selection macro-average | `recap_report_tables.py` — Eq. 51-52 (macro-avg validation BLEU/chrF++ per model) |
| Table 7 | Main results (headline) | `recap_evaluate.py` — main-matrix conditions (Stage 9 point 1) |
| Table 8 | Cumulative preference-construction ablation | `recap_evaluate.py` — ablation presets (Stage 9 point 2) |
| Table 9 | Candidate-diversity / pair-strategy comparison | `recap_evaluate.py` + targeted-ablation presets (Stage 9 point 3) |
| Table 10 | Human validation of preferences (500-pair subset) | `recap_sample_for_human_eval.py` (balanced 500-pair sample nikalta hai) → manual annotation → `recap_human_eval_analysis.py` (agreement %, Spearman ρ compute karta hai) |
| Table 11 | Appendix reporting checklist | Manual/static — meta-list hai, data nahi |
| Figure 1 | Reward-margin distribution (before/after filtering) | `recap_report_plots.py` — `pairs_all.csv` vs `pairs_balanced.csv` ke Δ distributions |
| Figure 2 | Model-participation bars (C_m+/C_m-) | `recap_report_plots.py` — Stage 5 ke balancing-stats se |
| Figure 3 | Heatmap of ΔBLEU/ΔChrF++/ΔCOMET per direction | `recap_report_plots.py` — Table 7 ke Δ-metrics se |

**Do naye files is stage ke liye:**

- `recap_report_tables.py` — sab `recap_eval/*/report.json` + `recap_pairs/*/*.csv`
  + training logs padhke Tables 3-9 (CSV/Markdown) emit karta hai.
- `recap_report_plots.py` — Figures 1-3 emit karta hai (PNG/SVG).
- `recap_sample_for_human_eval.py` — Table 10 ke liye 500-pair balanced subset
  sample karta hai (6 directions se, `pairs_balanced.csv` se) human annotation ke
  liye.
- `recap_human_eval_analysis.py` — human-labels wapas aane ke baad agreement % aur
  Spearman ρ compute karke Table 10 banata hai (ye do-step process hai — sample
  pehle, analysis baad mein, kyunki beech mein actual human annotation chahiye).

Sab reporting scripts sirf **already-saved outputs padhte hain** (eval reports,
pair files, logs) — kabhi khud se training/inference dobara nahi chalate, taaki
"paper table banana" aur "model train karna" do independent, resumable steps rahein.

---

## Summary (ek line mein pura pipeline)

```
maha_data_2 CSVs
  → train/val/test split
  → calibration stats
  → per-candidate reward
  → preference pairs (filter + balance)
  → DPO training (sft / dpo_raw / dpo_quality_only / dpo_no_confidence / recap_dpo)
  → optional GRPO refinement (sft_grpo / recap_dpo_grpo) + PPO baseline (sft_ppo)
  → test-set evaluation — main matrix + preference-ablation + targeted ablations
  → recap_infer.py (single reusable inference entrypoint)
  → recap_report_tables.py / recap_report_plots.py (Tables 3-9, Figures 1-3)
```

Har (language, direction, experiment, seed) combination is pura pipeline ko
independently follow karega — abhi total 6 (lang,direction) baar
(Bhili/Gondi/Mundari × hi2tgt/tgt2hi), aur ye list `config.py` se aati hai, kahin
aur se nahi — isliye naya language/direction/ablation add karna sirf ek
config-edit hai, code-edit nahi. Reproducibility (seeds, run manifests) aur
deadlock-free DDP dono is pipeline ke DPO/GRPO/PPO training stages mein built-in
hain, alag se sochne ki zaroorat nahi.
