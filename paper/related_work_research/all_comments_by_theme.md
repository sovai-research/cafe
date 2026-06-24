# All harvested reviewer comments (grouped by theme)

## weak baselines (32)

- **[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation]** _(major)_ Reviewer 2d7W: missing comparisons to standard probabilistic baselines (Multi-task Gaussian Processes, Latent/Neural ODE, mTANs).
- **[Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural Networks (GRIN)]** _(major)_ Reviewer 9yUs and public commenter (Lijun Sun): missing comparisons to low-rank matrix-factorization / low-rank tensor methods (with spatial/temporal regularization) and kriging/graph-signal interpolation baselines (Transdim framework).
- **[Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural Networks (GRIN)]** _(major)_ Reviewer JPtB: insufficient baselines; add NAOMI, E2GAN, kNN, graph-aware matrix factorization, kriging, and other graph-RNN architectures.
- **[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation]** _(major)_ Missing comparisons against relevant probabilistic / continuous-time baselines: Gaussian processes, Neural ODEs, and mTANs.
- **[Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models (SSSD)]** _(major)_ Baseline selection is inconsistent and under-motivated: strong baselines (NAOMI, NRTSI, SAITS, Autoformer) appear only in selected experiments; reviewer expects the strongest baselines to be run in every experiment where they plausibly apply.
- **[DiffImp: Efficient Diffusion Model for Probabilistic Time Series Imputation with Bidirectional Mamba Backbone]** _(major)_ Beating only other diffusion baselines is insufficient; reviewers demanded comparison to broader imputation methods (GNN-based GRIN, GATGPT, sparse spatiotemporal reconstruction) and standard benchmark datasets (PhysioNet 2012 healthcare, air quality).
- **[LSCD: Lomb-Scargle Conditioned Diffusion for Time Series Imputation]** _(major)_ Empirical gains over the non-spectral diffusion baseline (CSDI) are not significant in the main results table; the spectral component's benefit is not clearly demonstrated.
- **[HDDI: A Historical Data-Based Diffusion Imputation Method for High Missing Rate and Long-Term Gap Multivariate Time Series]** _(major)_ Baseline set is outdated (mostly pre-2021, only five methods) and omits strong recent imputers; reviewers asked for TimesNet, iTransformer, and notably PriSTI (CSDI+GNN) and GRIN given the long-gap / failed-node setting the paper targets.
- **[Self-attention-based Diffusion Model for Time-series Imputation in Partial Blackout Scenarios (SADI)]** _(minor)_ Traditional Bayesian/latent-variable generative imputation methods are not discussed; the term 'generative model' is used to mean only deep models, ignoring the broader class; the proposed random-masking 'solution' does not clearly match the missing-pattern criticism raised in the introduction.
- **[Online Low Rank Matrix Completion]** _(major)_ Experiments lack comparison against other recommendation-system / offline algorithms beyond a single UCB baseline.
- **[not-MIWAE: Deep Generative Modelling with Missing not at Random Data]** _(major)_ GAN-based imputers (e.g. GAIN) are insufficiently covered as comparison methods, despite their flexible parametric potential; more head-to-head comparison requested.
- **[MissDiff: Training Diffusion Models on Tabular Data with Missing Values]** _(major)_ A strong classical baseline, MICE/MissForest, is missing; reviewers also want clearer distinction between MICE-Forest and true MissForest implementations.
- **[DiffPuter: Empowering Diffusion Models for Missing Data Imputation]** _(major)_ Hyperparameter selection/tuning for baselines lacks transparency, raising fairness concerns; reviewers want the tuning protocol for all baselines detailed.
- **[M3-Impute: Mask-guided Representation Learning for Missing Value Imputation]** _(major)_ Program chairs rejected primarily over inadequate baseline hyperparameter optimization: several baselines (SVD, MIWAE, MIRACLE) performed worse than mean imputation, indicating unfair comparisons; tuning for MIWAE and others must be clarified.
- **[Relational Conformal Prediction for Correlated Time Series (CoREL)]** _(major)_ Reviewer asked for a wider set of baselines and a clearer rationale for which CP methods were selected; comparisons were only to SCP, SeqCP, SPCI/SCPI and the method should be benchmarked against more existing conformal approaches.
- **[Copula Conformal Prediction for Multi-step Time Series Forecasting (CopulaCPTS)]** _(major)_ Performance is very close to Messoudi et al.'s prior copula CP (which was not even designed for this setting), and Copula's results on real data were not reported; reviewers requested additional/stronger baselines and full reporting to show a genuine improvement.
- **[Conformal PID Control for Time Series Prediction]** _(major)_ The only baseline is ACI (plus ablations of the method itself); reviewers repeatedly asked why stronger time-series CP baselines such as EnbPI and SPCI (Xu & Xie) — which also achieve longitudinal coverage without infinite-width sets — were excluded, and how the method compares to them empirically.
- **[ProbTS: A Unified Toolkit to Probe Deep Time-series Forecasting]** _(major)_ Reviewers stressed the absence of naive / classical statistical probabilistic baselines (ETS and similar) — described as a real gap in the deep-learning literature — and the absence of strong simple baselines like feature-engineered XGBoost, which is 'quite hard to beat' on these datasets; the meta-review noted this was raised and only partially addressed.
- **[TSI-Bench: Benchmarking Time Series Imputation]** _(major)_ Reviewers (mYVZ, uzuS) said the method coverage is incomplete and the generative baselines (VAE/GAN/Diffusion) are outdated; they demanded inclusion of recent probabilistic/graph imputers (TimeCIB, DSPD-GP, mTANs, GRIN, TIDER, BiTGraph) or an explicit justification for the chosen subset, since for a benchmark paper comprehensiveness is critical.
- **[GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation]** _(minor)_ Reviewers asked for missing strong probabilistic baselines (TimeGrad, CSDI, GRU-NVP/RevIN) and recent foundation models (Timer, UniTS), and for comparison against existing benchmark toolkits (gluonts, pytorch-ts, ProbTS) in the related-work table.
- **[ProbTS: A Unified Toolkit to Probe Deep Time-series Forecasting]** _(major)_ Reviewers demanded naive/statistical baselines (ETS and other classical probabilistic models) and simple machine-learning baselines (XGBoost with hand-crafted features), noting these are hard to beat on the chosen datasets and their absence is a real gap in deep-learning probabilistic forecasting evaluation.
- **[Benchmarking Missing Data Imputation Methods in Socioeconomic Surveys]** _(minor)_ Reviewers asked for a naive 'no-imputation' baseline and methods that natively handle missingness (e.g. LightGBM), and justification for the chosen downstream models (RF/XGBoost) versus linear/neural alternatives; also asked why graph-based imputers (GRAPE, IGRM) were excluded.
- **[TSI-Bench: Benchmarking Time Series Imputation]** _(major)_ Benchmark omits recent generative/probabilistic imputers (TimeCIB, GRIN, DSPD-GP, TIDER); the included VAE/GAN/Diffusion implementations are outdated and not state-of-the-art.
- **[TimeFlow: Implicit Neural Representations for Time Series (Continuous Modeling for Imputation and Forecasting)]** _(major)_ Questioned whether baselines were given adequate hyperparameter tuning; asked for learning curves to confirm convergence and tests across different architecture sizes.
- **[Optimal Transfer Learning for Missing Not-at-Random Matrix Completion]** _(major)_ Comparisons need non-transfer MNAR methods and dedicated MNAR imputers (e.g. not-MIWAE), plus graph-based and VAE-based matrix completion baselines, to contextualize the minimax bounds.
- **[Controllable Text Generation with Identifiability Guarantee (Counterfactual Generation with Identifiability Guarantees)]** _(major)_ Baselines are dated (2019-2020, e.g. Optimus/GPT-2); reviewers requested comparison to modern LLMs (GPT-3.5) and a simple antonym-replacement baseline.
- **[Filling the Gaps: Multivariate Time Series Imputation by Graph Neural Networks (GRIN)]** _(major)_ Missing comparisons to low-rank matrix factorization methods, and especially MF variants that exploit graph/temporal side information (Rao 2015, Yu/TRMF 2016); low-rank models underperforming is not convincing given the inherently low-rank structure of traffic data.
- **[Filling the Gaps: Multivariate Time Series Imputation by Graph Neural Networks (GRIN)]** _(major)_ Baseline set is incomplete: should add NAOMI, E2GAN, kNN, kriging/GNN-kriging (Wu 2021), ST-MVL, and dynamic spatiotemporal GCNN imputation methods (Liang 2021).
- **[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation]** _(major)_ Missing probabilistic baselines: Multi-task Gaussian Processes, Latent ODE, mTANs; incomplete benchmarking against ODE-based and irregular-sampling methods.
- **[TSI-Bench: Benchmarking Time Series Imputation]** _(major)_ Missing recent generative/probabilistic methods (TimeCIB) and recent imputers (GRIN, TIDER, BiTGraph); the generative models covered (VAE/GAN/Diffusion) are already outdated.
- **[Time Series Continuous Modeling for Imputation and Forecasting with Implicit Neural Representations (TimeFlow)]** _(major)_ Requested simple imputation baselines (linear interpolation MAE, kNN) and that forecasting comparisons include imputation baselines; results merely match rather than exceed baselines like PatchTST.
- **[BayOTIDE: Bayesian Online Multivariate Time Series Imputation with Functional Decomposition]** _(major)_ Weak/insufficient baselines: lacks univariate probabilistic models, a single state-space model with Kalman filtering, and multi-output GP with linear+periodic kernels as natural comparisons.

## reproducibility (24)

- **[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation]** _(minor)_ Reviewer 8rwP: move DDPM definition and padding/masking details from appendix into the main text for clarity.
- **[Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models (SSSD)]** _(minor)_ Reviewer vTws & kCUh: insufficient implementation detail — loss function, batching, architecture, and whether all models support backward imputation are not clearly described; missing state-space-hybrid references (Kalman VAE, Deep Kalman Filters).
- **[Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models (SSSD)]** _(major)_ Model and training under-specified: reviewers could not concretely understand how state-space components are used, what loss/batching procedure was used, or how imputation is performed at run-time; requested all architectural components laid out in equation form.
- **[DiffImp: Efficient Diffusion Model for Probabilistic Time Series Imputation with Bidirectional Mamba Backbone]** _(minor)_ The masking/training strategy and the conditioning signal are not clearly explained: how masks are constructed when data already has/has-no missing values, and how exactly the conditional diffusion conditions are formed and used.
- **[LSCD: Lomb-Scargle Conditioned Diffusion for Time Series Imputation]** _(minor)_ Code was not actually available at the provided anonymous link; low-resolution figure math; missing related work on irregularly-sampled time series for regression/imputation/forecasting tasks.
- **[HDDI: A Historical Data-Based Diffusion Imputation Method for High Missing Rate and Long-Term Gap Multivariate Time Series]** _(major)_ Fairness/reproducibility concerns: unclear whether baseline hyperparameters were tuned; baselines absent from the repo so unclear if official code or a unified library (PyPOTS, Time-Series-Library) was used; only one dataset-processing script provided; environment/package versions not listed; standard deviations over the 5 trials omitted.
- **[Self-attention-based Diffusion Model for Time-series Imputation in Partial Blackout Scenarios (SADI)]** _(minor)_ Insufficient interpretation/justification of architecture design and the two-stage loss (eq.16); unclear amount of training data for the synthetic experiments; request to test non-Markov synthetic series rather than only the Markov construction.
- **[Online Low Rank Matrix Completion]** _(minor)_ Code was not provided, so reproducibility is unknown.
- **[Simplified Mamba with Disentangled Dependency Encoding for Long-Term Time Series Forecasting (SAMBA)]** _(major)_ A reviewer flagged that reported iTransformer MSE values contained errors and that CARD results inexplicably differed from the original paper while other methods matched identically, demanding clarification of baseline implementation sources and careful re-verification of all numbers.
- **[S7: Selective and Simplified State Space Layers for Sequence Modeling]** _(minor)_ Reviewers asked whether code is publicly available and noted missing definitions (how Lambda_k is computed) and inconsistent notation that hampered reproduction.
- **[S4M: S4 for Multivariate Time Series Forecasting with Missing Values]** _(major)_ Source code was not provided (low reproducibility); a reviewer also pressed whether baselines were run via official code or a unified library (Time-Series-Library, PyPOTS) since per-method data processing differs and unified interfaces are needed for fair comparison.
- **[not-MIWAE: Deep Generative Modelling with Missing not at Random Data]** _(minor)_ Notation is inconsistent (subscript i dropped after definition), hurting clarity.
- **[Identifiable Generative Models for Missing Not at Random Data Imputation (GINA)]** _(minor)_ Notation is unconventional/dense (capitals for sets, undefined 'auxiliary variables'), the GINA algorithm requires reading cited papers to follow, and the limitations discussion is only two sentences.
- **[Relational Conformal Prediction for Correlated Time Series (CoREL)]** _(minor)_ Reviewers wanted clearer statement of assumptions about the structure/nature of the underlying series, more detail on the fine-tuning/adaptation procedure, and a missing citation to Romano et al. (Conformalized Quantile Regression).
- **[Conformal PID Control for Time Series Prediction]** _(minor)_ Reviewers asked for the method's pseudocode in the paper, an environment.yml/datasets/figure-reproduction instructions in the code, and clearer definitions of terms used without definition (exchangeability, 'sharpest', burn-in period).
- **[TSI-Bench: Benchmarking Time Series Imputation]** _(major)_ Reviewer uzuS demanded complete per-method, per-dataset, per-task hyperparameter settings (e.g. which GP kernels, which diffusion noise schedules, whether optimal settings were sought on new datasets), noting their absence undermines trust in the rankings.
- **[ProbTS: A Unified Toolkit to Probe Deep Time-series Forecasting]** _(major)_ Reviewers asked whether hyperparameters/preprocessing were tuned individually per method (and to report the protocol in text), whether standard tuning packages (raytune) were used, how patch sizes were chosen, and whether reproduced results match each method's originally reported numbers.
- **[Benchmarking Missing Data Imputation Methods in Socioeconomic Surveys]** _(major)_ A reviewer demanded the authors show a third party can reproduce a substantial subset of experiments using only publicly available components (a key issue since one core dataset is proprietary).
- **[TSI-Bench: Benchmarking Time Series Imputation]** _(major)_ Missing hyperparameter documentation (GP kernels, diffusion noise schedules, timestep counts); reviewers questioned whether modern methods underperform simply because they were not tuned on the new datasets.
- **[TSI-Bench: Benchmarking Time Series Imputation]** _(minor)_ Library is hard to use outside its own framework: poor code organization, hardcoded missingness rates and algorithm lists, no clear input/output spec, and no support for static/spatial covariates. Author names leaked in supplementary code (anonymization breach).
- **[TimeFlow: Implicit Neural Representations for Time Series (Continuous Modeling for Imputation and Forecasting)]** _(minor)_ Requested explicit pointers to hyperparameter appendices and code release so baseline tuning could be replicated.
- **[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation]** _(minor)_ Important technical details (mu_DDPM definition, padding) relegated to the appendix; missing references to prior diffusion-for-sequence work.
- **[TSI-Bench: Benchmarking Time Series Imputation]** _(major)_ No hyperparameter documentation for the deep methods; unclear whether models were optimally tuned per dataset — suspicious that old BRITS beats modern methods, suggesting insufficient tuning rather than a real result.
- **[BayOTIDE: Bayesian Online Multivariate Time Series Imputation with Functional Decomposition]** _(major)_ Pervasive presentation problems: typos, inconsistent scalar/vector/matrix notation, improper LaTeX, missing RTS smoother formulas, and no model-size comparison vs CSDI.

## ablation (21)

- **[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation]** _(major)_ Reviewer yacf: the importance of the attention mechanism is unclear without comparison; asked for an ablation pitting the proposed attention against Bi-RNN, dilated convolution and flatten/no-temporal/no-feature variants.
- **[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation]** _(minor)_ Reviewer 2d7W: lacks ablation isolating the contribution of the temporal vs feature transformer layers.
- **[Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations (SPIN)]** _(minor)_ Reviewer 5BtN: requested ablation removing the skip-connection module, and moving ablations into the main text to show component importance.
- **[TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis]** _(major)_ Reviewer gBbZ: lacks an ablation on the efficacy of the adaptive aggregation; unclear whether frequency analysis is per-instance or per-batch.
- **[Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models (SSSD)]** _(major)_ Reviewer kCUh: S4-layer ablations limited to a single dataset/scenario; provide ablations across RM, MNR and BM at multiple missingness ratios.
- **[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation]** _(major)_ The importance of the two-dimensional (temporal + feature) attention mechanism is unclear without an ablation; reviewers asked for ablations isolating the attention design against alternatives (RNNs, dilated convolutions).
- **[Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models (SSSD)]** _(major)_ S4-layer ablation was shown only for one setting (20% BM on PTB-XL); reviewer requested ablations across multiple missingness types (RM, MNR) and ratios. Also requested study of diffusion-step count vs accuracy (inference-speed/accuracy tradeoff) and of train/test missingness-ratio mismatch generalization.
- **[Self-attention-based Diffusion Model for Time-series Imputation in Partial Blackout Scenarios (SADI)]** _(major)_ Decision-level concern that the main weaknesses were not resolved in rebuttal: drastic performance deterioration in the ablation setting, and the absence of training under the partial-blackout regime itself.
- **[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation]** _(minor)_ Reviewer asks for an ablation isolating the contribution of the temporal vs feature transformer layers.
- **[MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting]** _(major)_ No ablation directly comparing the modified Temporal Mamba Block against the original Mamba block; reviewers wanted that isolation.
- **[S4M: S4 for Multivariate Time Series Forecasting with Missing Values]** _(minor)_ Reviewers wanted a cleaner ablation isolating the prototype bank from masking (mask-only vs prototype-only vs both), a no-missing-values control to see if the prototype bank helps even with complete data, and sensitivity to prototype-bank initialization.
- **[DiffPuter: Empowering Diffusion Models for Missing Data Imputation]** _(major)_ No ablation replacing the diffusion model with other deep generative models (MIWAE, HI-VAE, VAEM, HH-VAEM) inside the same EM loop, so the diffusion-specific benefit is unclear; baselines also appear to use single-step rather than iterative EM imputation.
- **[Identifiable Generative Models for Missing Not at Random Data Imputation (GINA)]** _(major)_ Hard to isolate the benefit of identifiability from model misspecification; reviewers request experiments that separate identifiability effects from misspecification, and a sensitivity study over partition numbers S and L.
- **[M3-Impute: Mask-guided Representation Learning for Missing Value Imputation]** _(minor)_ Peer-set construction and stability are unclear, peer-size selection lacks guidance, ablation setups conflict, and tests with different GNN architectures are requested.
- **[Conformal PID Control for Time Series Prediction]** _(minor)_ A reviewer argued the 'scorecasting' (D) block is unnecessary/redundant — correcting for a mis-specified forecaster shouldn't be part of a CP framework and seems included mainly to force the PID analogy — and asked when scorecasting helps versus simply adapting the underlying model.
- **[This Time is Different: An Observability Perspective on Time Series Foundation Models (Toto + BOOM)]** _(major)_ Reviewers WAXQ & RNqe: ablations are overly simplistic -- merely removing a component is insufficient to show its role; missing parameter-sensitivity analysis and deeper component analysis.
- **[TimeFlow: Implicit Neural Representations for Time Series (Continuous Modeling for Imputation and Forecasting)]** _(minor)_ Reviewers wanted crisp, self-contained separate statements/results for imputation vs forecasting rather than merged descriptions, plus ablations justifying design choices (inference gradient steps, loss terms).
- **[Causal Counterfactual Regression Over Time via Representation Learning (Causal Contrastive Learning for Counterfactual Regression Over Time)]** _(major)_ Ablation showed removing the InfoNCE/contrastive loss had non-significant effect (esp. on low-dimensional synthetic covariates), undercutting the claimed contribution; reviewer asked whether theory is actually predictive of method effectiveness or detached from it.
- **[Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations (SPIN)]** _(minor)_ Skip-connection module needs an ablation; connections to forecasting tasks and downstream utility should be expanded.
- **[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation]** _(major)_ Lacks ablations isolating the contribution of the temporal/feature transformer attention vs simpler alternatives (no-temporal, no-feature, flattened, Bi-RNN, dilated conv); unclear the attention truly drives the gains.
- **[Time Series Continuous Modeling for Imputation and Forecasting with Implicit Neural Representations (TimeFlow)]** _(minor)_ Incomplete ablations: latent dimension and gradient-step sensitivity only run on one dataset (Electricity); wants ablations across all datasets and horizons.

## novelty (16)

- **[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation]** _(major)_ Reviewer yacf: methodological contribution is limited; the modified denoising function is a straightforward, minimal extension of DDPM.
- **[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation]** _(major)_ Reviewer 8rwP: lacks significant novelty; essentially an application of two existing methods (score matching + self-supervised masking already used in audio/NLP/CV inpainting); missing reference to prior inpainting work.
- **[Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural Networks (GRIN)]** _(major)_ Reviewer 9yUs: combining time series and graphs is not new; RNN+graph architectures already exist, so novelty is limited.
- **[Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations (SPIN)]** _(major)_ Reviewer seag (1/10) and AC meta-review: novelty is really limited — 'one more way to learn spatio-temporal dependencies via attention'; a GNN over (node, timestep) pairs with attention has been explored before.
- **[TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis]** _(major)_ Reviewer gBbZ: incorporating trend/seasonality component information 'is not a new idea'; technical novelty rated 2/5.
- **[TSI-Bench: Benchmarking Time Series Imputation]** _(major)_ Reviewer Evcs: 'Nothing novel is provided. All the methods and datasets are from existing works'; findings (missing patterns matter, forecasting architectures work for imputation, imputation helps downstream) are unsurprising; better as a journal review.
- **[Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models (SSSD)]** _(minor)_ Reviewer 5Tb4: the combination of DiffWave and S4 is 'slightly incremental'.
- **[not-MIWAE: Deep Generative Modelling with Missing not at Random Data]** _(major)_ Seen as a relatively straightforward extension of MIWAE using standard VAE/variational-bound/Monte-Carlo/reparameterization machinery; technical depth and novelty insufficiently highlighted.
- **[MissDiff: Training Diffusion Models on Tabular Data with Missing Values]** _(major)_ High-level similarity to misGAN, GAIN, and Jolicoeur-Martineau et al. is not adequately differentiated; main distinction seems to be only adapting the objective to neural nets.
- **[M3-Impute: Mask-guided Representation Learning for Missing Value Imputation]** _(major)_ Unclear how the approach differs from simply concatenating masks to the data, and how its initialization refinements differ from GRAPE/IGRM; reviewers ask for explicit differentiation.
- **[Filling the Gaps: Multivariate Time Series Imputation by Graph Neural Networks (GRIN)]** _(minor)_ Limited novelty in combining RNNs and graphs; architecture resembles prior graph-based RNN work, and the necessity of the two-stage decoder and bidirectional processing is not well justified.
- **[Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations (SPIN)]** _(major)_ Three of four reviewers and the AC saw limited novelty: attention over space-time product graphs where nodes are (node,timestep) pairs has been explored before; it is 'one more way to learn spatio-temporal dependencies via attention'.
- **[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation]** _(major)_ Seen as a straightforward/minimal extension of DDPM and essentially an application of two existing ideas (self-supervised masked training + diffusion); random-mask self-supervision and inpainting masks are standard.
- **[TSI-Bench: Benchmarking Time Series Imputation]** _(major)_ A reviewer argued a benchmark paper with no new methods/datasets reads as a journal survey rather than a conference contribution, and the findings (missing patterns matter, forecasting nets work for imputation, imputation helps downstream) are unsurprising.
- **[Time Series Continuous Modeling for Imputation and Forecasting with Implicit Neural Representations (TimeFlow)]** _(major)_ Limited novelty: applies known INR/meta-learning techniques (Dupont/Functa 2022, latent modulation) to time series with an architecture not specifically tailored to multivariate series.
- **[BayOTIDE: Bayesian Online Multivariate Time Series Imputation with Functional Decomposition]** _(minor)_ Methodological novelty over existing state-space GP / low-rank GP approximations seen as limited; asked why Matern kernel specifically and whether other low-rank GP approximations plug in.

## more datasets (14)

- **[Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models (SSSD)]** _(minor)_ All models should be evaluated on a common, consistent set of benchmarks for fair cross-experiment comparison.
- **[DiffImp: Efficient Diffusion Model for Probabilistic Time Series Imputation with Bidirectional Mamba Backbone]** _(major)_ Only two datasets (Electricity, MuJoCo) used for imputation; the ECG experiment central to SSSD was omitted despite replicating SSSD's other experiments. Reviewers also asked why missing rates differed by dataset (70/80/90% MuJoCo vs 10/30/50% Electricity) instead of a consistent grid, and why baselines weren't run on all datasets.
- **[LSCD: Lomb-Scargle Conditioned Diffusion for Time Series Imputation]** _(major)_ Experimental design unconvincing: synthetic sine data unfairly favors a sinusoidal-assumption method, and real-world irregular series often lack the seasonal structure the method exploits; reviewer questioned whether irregular-series imputation (esp. on medical Physionet data, where missingness is informative of sampling schedule) is even well-motivated.
- **[Simplified Mamba with Disentangled Dependency Encoding for Long-Term Time Series Forecasting (SAMBA)]** _(major)_ Reviewers asked for validation across more datasets (Exchange, Traffic, Weather, ECL, Solar) because key claims (e.g. Table 3 / activation removal) were demonstrated on too few or inappropriate datasets, and the chosen Exchange dataset was deemed unsuitable for the claim.
- **[not-MIWAE: Deep Generative Modelling with Missing not at Random Data]** _(minor)_ More synthetic experiments with controlled parameters are recommended to probe behaviour when the missingness mechanism is known vs unknown.
- **[MissDiff: Training Diffusion Models on Tabular Data with Missing Values]** _(major)_ Evaluation should move to the larger ForestDiffusion 27-dataset benchmark rather than the smaller original set, to give more diverse/plausible results and compare against current diffusion/autoregression SOTA.
- **[Identifiable Generative Models for Missing Not at Random Data Imputation (GINA)]** _(minor)_ Add broader benchmarks (e.g. UCI datasets) for a more comprehensive comparison, and address estimation consistency under MNAR.
- **[Relational Conformal Prediction for Correlated Time Series (CoREL)]** _(major)_ Performance under genuinely volatile / strongly non-stationary conditions is not thoroughly evaluated despite the adaptivity claim; reviewers wanted the spatio-temporal value demonstrated on a real multivariate use case rather than only standard benchmarks, and the practical utility to an end user shown concretely.
- **[TSI-Bench: Benchmarking Time Series Imputation]** _(major)_ Downstream evaluation only uses XGBoost; reviewers wanted with/without-imputation comparisons across multiple downstream models and richer analysis of WHEN each method succeeds rather than just aggregate metrics.
- **[TimeFlow: Implicit Neural Representations for Time Series (Continuous Modeling for Imputation and Forecasting)]** _(major)_ All evaluation datasets exhibit strong hourly/daily seasonality, which structurally favors a Fourier/seasonal model; reviewer demanded evaluation on non-seasonal datasets (e.g. from forecastingdata.org) to show the gains are not an artifact of periodicity.
- **[Causal Counterfactual Regression Over Time via Representation Learning (Causal Contrastive Learning for Counterfactual Regression Over Time)]** _(major)_ Evaluation rests on a single synthetic dataset (five seeds) plus one semi-synthetic dataset; multiple reviewers demanded experiments on real datasets and across varied generative processes, sequence lengths, and high-dimensional covariates.
- **[Filling the Gaps: Multivariate Time Series Imputation by Graph Neural Networks (GRIN)]** _(major)_ Missing-data simulation is unrealistic (only 5% block / 25% point missing); requested sensitivity analysis at higher missing rates and on toy datasets with various missingness patterns.
- **[Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations (SPIN)]** _(major)_ Standard datasets (PEMS-BAY, AQI) are insufficient without novel analysis; reviewer wanted emerging/harder datasets and broader applicability beyond sensor networks.
- **[BayOTIDE: Bayesian Online Multivariate Time Series Imputation with Functional Decomposition]** _(major)_ Limited evaluation: only two observed ratios (50%/70%) tested; wants a sweep of missing rates (90/80/70/60/50%) and multiple missingness patterns beyond random.

## scalability (12)

- **[Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural Networks (GRIN)]** _(major)_ Reviewer 9yUs: adding message passing in the recurrent gates adds overhead; provide memory/runtime comparisons to baselines (no empirical cost evidence given).
- **[DiffImp: Efficient Diffusion Model for Probabilistic Time Series Imputation with Bidirectional Mamba Backbone]** _(major)_ The central claim is linear time complexity, yet runtime/time-cost is essentially absent from the experiments; reviewers found it 'unbelievable' that wall-clock time was omitted given the efficiency claim, and asked for training memory and parameter counts too.
- **[LSCD: Lomb-Scargle Conditioned Diffusion for Time Series Imputation]** _(major)_ Lomb-Scargle is more expensive than FFT; reviewers required an analysis of training and inference time / computational efficiency vs FFT, which was missing.
- **[HDDI: A Historical Data-Based Diffusion Imputation Method for High Missing Rate and Long-Term Gap Multivariate Time Series]** _(major)_ The sliding-window (step size 1) historical matching scales poorly / linearly with data size; reviewers repeatedly demanded a complexity analysis, window-length selection rationale, and real-time feasibility / production-scaling discussion that the paper lacked.
- **[Multi-Mode Deep Matrix and Tensor Factorization]** _(major)_ The tensors used in the experiments are quite small; reviewer asks how the proposed model scales to large tensors and notes that running time of the method and baselines is not reported.
- **[MissDiff: Training Diffusion Models on Tabular Data with Missing Values]** _(minor)_ Unclear how a single network handles the 2^d possible missing patterns without extraordinary capacity requirements.
- **[DiffPuter: Empowering Diffusion Models for Missing Data Imputation]** _(major)_ Limited evaluation on large/high-dimensional datasets; reviewers asked for tests on something like Covertype (200k+ rows, 50+ features).
- **[M3-Impute: Mask-guided Representation Learning for Missing Value Imputation]** _(major)_ The bipartite-graph size scales drastically with data dimensionality; reviewers want computational/space complexity analysis and tests on large high-dimensional datasets, plus a clear description of the inference procedure for new query instances.
- **[Relational Conformal Prediction for Correlated Time Series (CoREL)]** _(major)_ The method requires graph inference at every update step, which adds non-trivial overhead; runtime comparisons against standard CP methods are needed to substantiate the efficiency/scalability claims.
- **[Conformal PID Control for Time Series Prediction]** _(major)_ All datasets are small (largest ~3000 points); reviewers asked whether the method works unchanged in true streaming settings with data arriving every second, and requested an experiment on a very long time series.
- **[Filling the Gaps: Multivariate Time Series Imputation by Graph Neural Networks (GRIN)]** _(major)_ No runtime or memory comparison provided despite the overhead of message passing; needs computational complexity analysis or empirical scalability evidence.
- **[Time Series Continuous Modeling for Imputation and Forecasting with Implicit Neural Representations (TimeFlow)]** _(minor)_ Computational cost higher than transformers, and the train/test temporal-gap assumption (operational delay) is questioned as unrealistic for hourly data.

## calibration (11)

- **[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation]** _(minor)_ Reviewer 2d7W: why was negative log-likelihood (NLL) omitted when evaluating probabilistic interpolation? Requested NLL as an evaluation metric.
- **[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation]** _(minor)_ Missing probabilistic evaluation: reviewer wanted negative log-likelihood (NLL) metrics reported, not just point/CRPS-style metrics.
- **[MissDiff: Training Diffusion Models on Tabular Data with Missing Values]** _(minor)_ Tightness of the claimed likelihood upper bound (Theorem 3.2) is unquantified.
- **[Relational Conformal Prediction for Correlated Time Series (CoREL)]** _(major)_ Method is evaluated with coverage (gap between nominal and observed coverage), prediction-interval width, and Winkler score; reviewers said the model only clearly wins on Winkler score while other methods match or beat it on coverage and width, so superiority is not fully established.
- **[Copula Conformal Prediction for Multi-step Time Series Forecasting (CopulaCPTS)]** _(major)_ Reviewers emphasized the central claims are finite-sample validity (coverage) and efficiency (narrower intervals); meta-review noted opinions were mixed on whether the empirical interval gains over prior work are real, and asked the authors to lean on the validity guarantee + efficiency as the true contribution.
- **[Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting (TSDiff)]** _(minor)_ A reviewer praised the introduction of a fairer comparison metric but, more broadly, the evaluation centers on CRPS-style probabilistic accuracy; the panel's split scores (4/5/7/7) reflect unease about whether the probabilistic-quality claims are rigorously enough supported.
- **[TimeFlow: Implicit Neural Representations for Time Series (Continuous Modeling for Imputation and Forecasting)]** _(minor)_ Metric choice questioned: why MAE over MAPE (authors: normalized series make MAPE unstable near zero denominators).
- **[Optimal Transfer Learning for Missing Not-at-Random Matrix Completion]** _(minor)_ Reported only max squared error; reviewers required standard metrics (RMSE, MAE) across all tables.
- **[Controllable Text Generation with Identifiability Guarantee (Counterfactual Generation with Identifiability Guarantees)]** _(minor)_ Evaluation issues: single sentiment classifier may bias accuracy, BLEU penalizes valid word substitutions, anomalous accuracy ranges below random, no human evaluation, and missing mean/std reporting across runs.
- **[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation]** _(minor)_ Omits negative log-likelihood (NLL) as an uncertainty metric with no justification; reviewers wanted standard probabilistic evaluation (authors argued CRPS is more robust than schedule-dependent NLL).
- **[BayOTIDE: Bayesian Online Multivariate Time Series Imputation with Functional Decomposition]** _(major)_ All evaluation metrics are univariate despite the model being multivariate; should report multivariate uncertainty metrics (energy score, sum-CRPS) to show the multivariate-correlation benefit.

## MNAR (11)

- **[Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models (SSSD)]** _(major)_ Missingness nomenclature is non-standard and conflated with the literature: what the paper calls RM/MNR/BM/TF should be mapped to MCAR/MAR/MNAR; reviewer noted the paper does not actually consider true MNAR (missingness correlated with the unobserved value).
- **[DiffImp: Efficient Diffusion Model for Probabilistic Time Series Imputation with Bidirectional Mamba Backbone]** _(minor)_ Only synthetic (injected) missingness is tested; reviewers asked for real-world incomplete datasets to demonstrate effectiveness.
- **[not-MIWAE: Deep Generative Modelling with Missing not at Random Data]** _(major)_ The missingness process is modelled too simplistically and is not explored systematically across varying missing ratios (0-100%); reviewer wants the method stressed across the full missingness spectrum.
- **[MissDiff: Training Diffusion Models on Tabular Data with Missing Values]** _(major)_ The 'general framework for incomplete data' claim ignores the distinction between MCAR/MAR/MNAR mechanisms, which are crucial for statistical efficiency and identifiability; how theory changes under each mechanism is not discussed.
- **[DiffPuter: Empowering Diffusion Models for Missing Data Imputation]** _(major)_ Limited exploration of MAR and MNAR mechanisms in the main results; implementation details of the MAR/MNAR mask procedures are missing and should be in the main text.
- **[M3-Impute: Mask-guided Representation Learning for Missing Value Imputation]** _(major)_ Insufficient discussion of robustness across missingness types (MCAR/MAR/MNAR) and missing rates, and why the method works better on some datasets than others (e.g. weak on highly independent / extreme-correlation datasets like KIN8NM, NAVAL).
- **[TSI-Bench: Benchmarking Time Series Imputation]** _(major)_ Reviewer mYVZ said the benchmark insufficiently explores diverse missingness scenarios (MNAR/MCAR/periodic patterns) and asked how pre-existing missing values in the raw datasets were handled during training and testing.
- **[Benchmarking Missing Data Imputation Methods in Socioeconomic Surveys]** _(major)_ A reviewer asked the authors to move the missingness-generation mechanism into the main text: the logistic-model procedure for MAR/MNAR, how the parameters beta are learned, whether the same logistic models are reused or re-estimated per dataset, and how temporary imputations work under MNAR — calling this central to the benchmark's validity.
- **[TSI-Bench: Benchmarking Time Series Imputation]** _(major)_ Missingness scenarios are too limited: lacks MNAR (and clearer MCAR) settings and periodic/structured missingness patterns; pre-existing missing values in the raw data are not adequately handled.
- **[Optimal Transfer Learning for Missing Not-at-Random Matrix Completion]** _(major)_ Unclear whether experiments use real missingness or synthetic masking, and whether ground-truth values even exist for the masked entries in real datasets; reviewer wanted evaluation on datasets where entry recovery is the genuine target.
- **[TSI-Bench: Benchmarking Time Series Imputation]** _(major)_ Limited missingness scenarios: lacks MNAR/MCAR distinctions and periodic missing patterns; only basic settings explored.

## identifiability (10)

- **[not-MIWAE: Deep Generative Modelling with Missing not at Random Data]** _(major)_ It is unclear how prior information about the missingness is formally specified and leveraged, and convergence claims lack rigorous proof.
- **[Identifiable Generative Models for Missing Not at Random Data Imputation (GINA)]** _(major)_ The identifiability obtained is parametric, not nonparametric, and this distinction must be made explicit; reviewers also want clarity on whether identifiability is necessary or sufficient for unbiased imputation.
- **[Identifiable Generative Models for Missing Not at Random Data Imputation (GINA)]** _(major)_ There is no proof in the appendix that the proposed GINA algorithm itself satisfies assumption A4, and the neural-network conditions for the decoder f remain unclear.
- **[Learning Treatment Effects in Panels with General Intervention Patterns]** _(major)_ Main theorems are derived assuming a single treatment effect (k=1); reviewer pressed whether generalizing to multiple treatment effects (k>1) is just notational or involves real technical barriers, and whether Assumption 3 must then hold for every linear combination of treatments.
- **[Learning Treatment Effects in Panels with General Intervention Patterns]** _(major)_ Reviewer asked whether distinct treatment effects could cancel in the tangent space, which would make the effect non-identifiable under the stated assumptions.
- **[Optimal Transfer Learning for Missing Not-at-Random Matrix Completion]** _(major)_ The structured-transfer assumption (Def 1.2) is restrictive; reviewer asked what happens when it fails, since real-world MNAR mechanisms are more complex than the assumed model.
- **[Causal Counterfactual Regression Over Time via Representation Learning (Causal Contrastive Learning for Counterfactual Regression Over Time)]** _(major)_ Reviewer questioned the sequential-ignorability / no-unobserved-confounding assumption and its robustness; authors had to show via masked-confounder ablation that the model still beats baselines at long horizons.
- **[Causal Counterfactual Regression Over Time via Representation Learning (Causal Contrastive Learning for Counterfactual Regression Over Time)]** _(major)_ Skepticism about the encoder invertibility claim used in the representation argument; reviewer demanded a formal proof rather than an appeal to disentangled-representation literature.
- **[Controllable Text Generation with Identifiability Guarantee (Counterfactual Generation with Identifiability Guarantees)]** _(major)_ Asked authors to pin down what 'identifiability guarantee' formally means (standard statistical identifiability) and to formalize the 'relative sparsity' notion used.
- **[BayOTIDE: Bayesian Online Multivariate Time Series Imputation with Functional Decomposition]** _(minor)_ Non-identifiability in the UV low-rank factorization is raised — how does it affect results, and how do GPs produce the linear-looking trend components?

## weak novelty (9)

- **[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation]** _(major)_ Methodological contribution is rather limited: the method is only a minimal modification of an existing DDPM; the technique combination is described as 'too incremental' and 'essentially applies existing methods without substantial innovation.'
- **[Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models (SSSD)]** _(minor)_ Combining two existing models (DiffWave + S4) is 'slightly incremental.'
- **[DiffImp: Efficient Diffusion Model for Probabilistic Time Series Imputation with Bidirectional Mamba Backbone]** _(major)_ Contribution is limited: the work reads as a combination of diffusion + Mamba; the bidirectional-SSM trick is not new (analogous to bi-LSTM), and no theoretical result supports the proposed techniques.
- **[LSCD: Lomb-Scargle Conditioned Diffusion for Time Series Imputation]** _(major)_ Novelty is limited: the framework is still mainly CSDI with small changes, and using spectral/periodogram information for time series is well-studied; no theoretical analysis of how the Lomb-Scargle periodogram affects the diffusion process.
- **[HDDI: A Historical Data-Based Diffusion Imputation Method for High Missing Rate and Long-Term Gap Multivariate Time Series]** _(major)_ Reviewer inspected the code and found the backbone (diff_HDDI) is exactly the CSDI denoiser, which is not disclosed; the only real contribution is a historical-data supplement module, judged insufficient. Incorporating historical data is also a long-standard time-series technique.
- **[Self-attention-based Diffusion Model for Time-series Imputation in Partial Blackout Scenarios (SADI)]** _(major)_ Claim of modeling feature and temporal dependency 'jointly' is not justified: like CSDI, the model applies feature and temporal blocks sequentially, so reviewers did not accept the 'joint' framing; novelty over existing diffusion imputers not clearly discussed.
- **[Simplified Mamba with Disentangled Dependency Encoding for Long-Term Time Series Forecasting (SAMBA)]** _(major)_ Reviewers said the contribution reduces to 'only removing the nonlinear activation functions of Mamba'; the disentangled-dependency framing was seen as not clearly differentiated from existing work, and the AC rejected partly because the paper failed to establish true novelty over prior research.
- **[S7: Selective and Simplified State Space Layers for Sequence Modeling]** _(major)_ Reviewers said adding input-dependent dynamics to S5 is conceptually the same thing Mamba (S6) already did to S4, and that the stable-reparameterization equations were borrowed directly from StableSSM, leaving the original contribution unclear.
- **[DiffImp: Efficient Diffusion Model for Probabilistic Time Series Imputation with Bidirectional Mamba Backbone]** _(major)_ Multiple reviewers viewed the work as a 'pieced-together' combination of an existing diffusion model and existing Mamba; bidirectional SSM and channel-dependency modeling are standard (cf. bi-LSTM), so novelty/contribution were judged limited, with no clear research question.

## downstream utility (7)

- **[Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural Networks (GRIN)]** _(minor)_ Reviewer 9yUs: unclear why explicit imputation is needed vs. just running a forecasting model; clarify the imputation-vs-downstream-prediction relationship.
- **[Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations (SPIN)]** _(minor)_ Reviewer 5BtN and AC: discuss how the imputation method helps other time-series tasks like forecasting; connection to forecasting/other tasks is missing.
- **[TSI-Bench: Benchmarking Time Series Imputation]** _(minor)_ Reviewer mYVZ: should evaluate downstream models with AND without imputation across multiple model types, not just XGBoost.
- **[DiffPuter: Empowering Diffusion Models for Missing Data Imputation]** _(major)_ Missing evaluation on downstream classification/regression tasks to show practical impact beyond RMSE/MAE imputation metrics.
- **[M3-Impute: Mask-guided Representation Learning for Missing Value Imputation]** _(minor)_ No downstream-task evaluation despite imputation being a preprocessing step; RMSE results and standard deviations are also missing from the main table.
- **[TSI-Bench: Benchmarking Time Series Imputation]** _(minor)_ Reviewer mYVZ said downstream evaluation is too limited (only XGBoost) and should compare with/without imputation across multiple downstream model types.
- **[TSI-Bench: Benchmarking Time Series Imputation]** _(minor)_ Downstream evaluation too thin (only XGBoost); should compare with vs without imputation across more tasks.

## leakage (7)

- **[Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models (SSSD)]** _(major)_ Reviewer vTws: concern about data-leakage risk from backward/forward smoothing imputation steps — discuss when backward smoothing introduces future information leakage. (Authors later acknowledged and corrected a Solar-dataset preprocessing leakage, revising test MSE.)
- **[Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models (SSSD)]** _(major)_ A data-leakage flaw was identified post-submission: train/test leakage in the Solar dataset preprocessing inflated results; authors had to retrain (MSE worsened from ~2.7e2 to ~5.03e2) and fix the public preprocessing code. (Raised by an external reader, conceded by authors.)
- **[Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models (SSSD)]** _(major)_ Reviewer asked whether using a backward/smoothing imputation step leaks future information into imputed data when imputation is a precursor to training downstream models, and in which applications a backward step is even justified.
- **[LSCD: Lomb-Scargle Conditioned Diffusion for Time Series Imputation]** _(major)_ Methodological correctness concern: the spectral consistency loss computes the 'ground-truth' spectrum from the masked observed condition rather than the original full series, and the figure's 'noisy imputation target' appears shifted rather than noise-distorted as in CSDI, suggesting a possible diffusion-process implementation error.
- **[DiffPuter: Empowering Diffusion Models for Missing Data Imputation]** _(minor)_ Possible train/test mask-distribution mismatch should be discussed.
- **[GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation]** _(major)_ A reviewer warned of test-data leakage: foundation models (TimesFM, Chronos) are evaluated from released checkpoints while their pretraining corpora overlap evaluation datasets (e.g. TimesFM pretrained on Electricity, which is also a test set), possibly explaining suspiciously strong scores; demanded re-training all foundation models on the leakage-controlled split for a fair comparison.
- **[Time Series Continuous Modeling for Imputation and Forecasting with Implicit Neural Representations (TimeFlow)]** _(major)_ Fairness concern: test-time auto-decoding updates the latent vector via optimization, which is equivalent to implicit fine-tuning / different INR parameters, whereas baselines use a single forward pass — an unfair comparison.

## honesty of claims (6)

- **[Learning Treatment Effects in Panels with General Intervention Patterns]** _(minor)_ Reviewer wanted the practical limitations of the asymptotic-normality result (Theorem 2) spelled out, since it relies on independence that the imputation theorem (Theorem 1) does not require.
- **[TSI-Bench: Benchmarking Time Series Imputation]** _(minor)_ Surprising result that an early method (BRITS) beats recent approaches was left unexplained; reviewers were puzzled and wanted the cause diagnosed rather than just reported.
- **[TimeFlow: Implicit Neural Representations for Time Series (Continuous Modeling for Imputation and Forecasting)]** _(major)_ Limited novelty beyond conditional INRs with modulations; reviewer disputed the 'meta-learning' framing, saying Algorithm 2 is really inference adapting sequence-specific parameters; authors added a Limitations section restricting claims to periodic data.
- **[Optimal Transfer Learning for Missing Not-at-Random Matrix Completion]** _(major)_ Method characterized as largely a combination of existing techniques; active-sampling scheme criticized as hard to implement in practice despite the paper's claims of realism.
- **[Causal Counterfactual Regression Over Time via Representation Learning (Causal Contrastive Learning for Counterfactual Regression Over Time)]** _(minor)_ Performance degrades as horizon tau grows and baselines win at short horizons; reviewers asked authors to explain the decline and explicitly state the short-horizon limitation.
- **[Controllable Text Generation with Identifiability Guarantee (Counterfactual Generation with Identifiability Guarantees)]** _(major)_ Introduction overstates generality (broad counterfactual generation) while empirics only cover sentiment transfer; reviewers flagged the scope inflation and mislabeling of sentiment as 'style'.

## more datasets / generality (4)

- **[Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural Networks (GRIN)]** _(minor)_ Reviewer hJvE: clarify applicability beyond sensor networks (e.g., financial, healthcare panels).
- **[Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations (SPIN)]** _(major)_ Reviewer ajb9 & seag & AC: experiments confined to sensor networks (PEMS-BAY etc.); need test cases beyond sensor networks and a clearer practical motivation.
- **[TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis]** _(minor)_ Reviewer Re8p: applicability unclear for multi-dataset settings with variable period lengths (e.g., patient hospital data).
- **[Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting (TSDiff)]** _(major)_ Experiments are univariate only; reviewers asked whether the method extends to multivariate time series and to other downstream tasks (classification, clustering, imputation with actually-missing training data, synthetic-augmented forecasting).

## missing baselines (4)

- **[S7: Selective and Simplified State Space Layers for Sequence Modeling]** _(major)_ Because S7 is built on S5, reviewers insisted it must always be compared against S5 (and against Mamba/Mamba2), but those comparisons were absent from several result tables.
- **[S4M: S4 for Multivariate Time Series Forecasting with Missing Values]** _(major)_ A reviewer faulted the absence of the irregular-sampled / missing-value forecasting literature (GRU-ODE-Bayes, GraFITi, CRU, latent-ODE/graph models) and their standard naturally-missing datasets, plus simple linear-interpolation + PatchTST / iTransformer baselines that may beat S4M.
- **[DiffImp: Efficient Diffusion Model for Probabilistic Time Series Imputation with Bidirectional Mamba Backbone]** _(major)_ Reviewers asked for comparison against more imputation methods beyond diffusion ones (GNN-based GRIN, GATGPT, sparse-observation spatiotemporal graph methods) rather than only diffusion baselines, and to run all baselines on all datasets instead of subsets.
- **[Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting (TSDiff)]** _(major)_ Because conditioning on observed timesteps is essentially an inpainting/inverse problem, reviewers asked that diffusion inverse-problem baselines (e.g. DDRM) be included or thoroughly discussed.

## runtime (4)

- **[DiffPuter: Empowering Diffusion Models for Missing Data Imputation]** _(major)_ Computational-cost / time-complexity analysis is relegated to the appendix; K EM iterations multiply training cost K-fold and this should be quantified and compared against TabCSDI and MissDiff in the main text.
- **[M3-Impute: Mask-guided Representation Learning for Missing Value Imputation]** _(major)_ No runtime / computational-resource comparison versus other methods; repeatedly requested.
- **[Copula Conformal Prediction for Multi-step Time Series Forecasting (CopulaCPTS)]** _(minor)_ Computational times are not reported; standard deviations come from only 3 runs, raising the worry that the algorithm is too slow to run more — reviewers asked for runtime reporting and more seeds.
- **[Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting (TSDiff)]** _(major)_ Reviewers repeatedly flagged the high computational/inference cost of the diffusion model and asked for explicit forecasting-speed comparisons against conditional diffusion models, plus clarification of whether the self-guidance optimum tau is expensive to compute per data point.

## limited novelty (3)

- **[TIDER: Multivariate Time-series Imputation with Disentangled Temporal Representations]** _(major)_ The approach is seen as a fairly simple combination of low-rank matrix factorization with seasonality/trend notions and may not be novel enough for the venue; disentangled factors are pre-defined rather than learned.
- **[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation]** _(minor)_ Reviewer views the method as a novel but incremental combination of existing techniques (conditional score matching / inpainting-style masking applied to time series), with individual components drawn from prior work.
- **[This Time is Different: An Observability Perspective on Time Series Foundation Models (Toto + BOOM)]** _(major)_ Reviewers WAXQ & RNqe: the core components (proportional/factorized variate+time attention, Student-t mixture head) already appear in prior foundation models (e.g. Student-t in MOIRAI); a fixed ratio balancing variate-wise vs time-wise attention is only incremental, and it is unclear how the components are specifically tailored to observability data versus being generic.

## mixed-type evaluation (3)

- **[not-MIWAE: Deep Generative Modelling with Missing not at Random Data]** _(minor)_ Unclear how discrete/categorical variables in mixed datasets are evaluated under the RMSE metric.
- **[DiffPuter: Empowering Diffusion Models for Missing Data Imputation]** _(minor)_ One-hot encoding of categoricals plus mean imputation is problematic and does not address multimodality; a multi-mode simulation study is requested.
- **[M3-Impute: Mask-guided Representation Learning for Missing Value Imputation]** _(major)_ Method cannot handle categorical features (a major limitation); reviewers want categorical/mixed-type support and a limitations section acknowledging it.

## assumptions (3)

- **[Identifiable Generative Models for Missing Not at Random Data Imputation (GINA)]** _(major)_ Assumptions A1-A3 appear strong and close to MAR; reviewers ask for concrete simple examples showing when they actually hold and what happens when the assumed model fails to fit the observed data.
- **[Learning Treatment Effects in Panels with General Intervention Patterns]** _(major)_ The independent-noise assumption (across both time steps and units) is unrealistic; in real panels noise is systematic and positively correlated across time and across experimental units, so the limitation should be made explicit for practitioners.
- **[Controllable Text Generation with Identifiability Guarantee (Counterfactual Generation with Identifiability Guarantees)]** _(major)_ Theoretical assumptions are internally tense (invertibility of Jacobians conflicts with sparsity constraints) and disconnected from the loose loss-based training implementation, so the guarantees may not transfer to the actual method.

## irregular sampling (2)

- **[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation]** _(minor)_ Reviewer 2d7W: discretizing irregularly-sampled data risks information loss.
- **[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation]** _(minor)_ Asked whether the method can directly handle irregularly sampled sequences and how interpolation vs imputation differ after PhysioNet discretization.

## MNAR / missingness patterns (2)

- **[Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural Networks (GRIN)]** _(major)_ Reviewer jZvK: missing-data simulation is too simple (only 5% block + 25% point missing); test higher missing rates and more patterns, and compare more extensively with BRITS.
- **[TSI-Bench: Benchmarking Time Series Imputation]** _(major)_ Reviewer mYVZ: lacks diverse missing-data scenarios (MNAR vs MCAR, periodic missingness) and deeper analysis of WHY methods excel in given situations; 'experimental design lacks depth'.

## dynamic graph (2)

- **[Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural Networks (GRIN)]** _(minor)_ Reviewer jZvK and ajb9: static adjacency-matrix assumption is unsuitable for dynamic relational data; address dynamic-graph capability. Also RNN core cannot handle irregularly-sampled data.
- **[Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations (SPIN)]** _(minor)_ Reviewer ajb9: static-graph assumption not addressed; graph built from pairwise distances is oversimplified (ignores e.g. sensor elevation).

## clarity (2)

- **[Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations (SPIN)]** _(minor)_ Reviewer fr5V & ajb9: insufficient explanation of sensor-level covariates and the d-dimensional measurement notation; Section 3 / hierarchical attention is too compact to follow.
- **[Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations (SPIN)]** _(minor)_ Hierarchical attention section too compressed and hard to follow; notation (M_t, X_<t, X_>t, N(i)) undefined; wants a diagram distinguishing SPIN vs SPIN-H and clearer contribution statement.

## scope of claim (2)

- **[TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis]** _(major)_ Reviewer Re8p & gBbZ: the periodicity assumption is problematic for non-strictly-periodic data; the 'general time series' claim may really only hold for periodic series — risks being positioned to a narrower scenario.
- **[There are no Champions in Long-Term Time Series Forecasting]** _(minor)_ A reviewer questioned whether general-purpose forecasting is even a meaningful target if domain-specific SOTA (e.g. 3D-ViT for weather, feature+tree ensembles for finance) always wins, challenging the value of the 'no champion' framing for general TSF.

## runtime / scalability (2)

- **[Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models (SSSD)]** _(minor)_ Reviewer kCUh: quantify the diffusion-timestep inference-speed vs accuracy tradeoff; channel-splitting for 100+ channels is unexplained.
- **[DiffImp: Efficient Diffusion Model for Probabilistic Time Series Imputation with Bidirectional Mamba Backbone]** _(major)_ The main selling point is linear complexity, yet several reviewers found time cost 'totally ignored' in experiments and demanded explicit runtime, training memory, parameter counts, and scalability experiments on additional large real-world datasets to substantiate the efficiency claim.

## motivation (2)

- **[DiffImp: Efficient Diffusion Model for Probabilistic Time Series Imputation with Bidirectional Mamba Backbone]** _(major)_ The motivation for modeling inter-channel dependence and for 'backbone compatibility' in diffusion is not established; it reads as an exploratory architecture search without a clear technical/theoretical research question, and the imputation+forecasting scope feels pieced together.
- **[Self-attention-based Diffusion Model for Time-series Imputation in Partial Blackout Scenarios (SADI)]** _(major)_ The link between the emphasized 'partial blackout' missingness concept and the actual method is unclear: partial blackout is discussed at length in intro/abstract but barely in the methodology section; reviewers were unconvinced how the architecture is essentially connected to that missing pattern or what advantage it gives over methods that already handle general missing positions.

## weak baselines / missing comparisons (2)

- **[Multi-Mode Deep Matrix and Tensor Factorization]** _(major)_ Recent nonlinear tensor-completion models (e.g., CoSTCo) are not discussed or compared; more recent methods should be added as baselines.
- **[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation]** _(major)_ Missing comparisons to standard probabilistic baselines: single-task and multi-task Gaussian Processes, neural-ODE methods (which should also appear in the forecasting table), and mTAN (which can produce probabilistic interpolation by sampling latents).

## justify design choice (2)

- **[Simplified Mamba with Disentangled Dependency Encoding for Long-Term Time Series Forecasting (SAMBA)]** _(major)_ Reviewers questioned why Mamba is preferred over self-attention for cross-variate dependency (self-attention called a 'perfect fit'), and whether preserving channel order-dependency is actually necessary, since recent Transformers do well without it.
- **[Is Bidirectionality Necessary in Mamba for Time Series Forecasting? (FSMamba)]** _(major)_ Reviewers questioned the motivation for applying Mamba (designed for temporal/sequential modeling) to unordered channel dependencies at all, and found the contradictory 1D-conv-removal result on order-bearing PEMS data inadequately explained.

## statistical rigor (2)

- **[MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting]** _(major)_ Results lacked error bars and the limited number of datasets made it hard to judge significance of the claimed gains; ETTh1 and ETTm1 were missing despite being standard benchmarks.
- **[BayOTIDE: Bayesian Online Multivariate Time Series Imputation with Functional Decomposition]** _(major)_ Single-run results with no Monte Carlo repetition or standard deviations; needs multiple runs with reported variance.

## scalability / runtime (2)

- **[S4M: S4 for Multivariate Time Series Forecasting with Missing Values]** _(major)_ Computational-efficiency comparison was missing; reviewers wanted training/inference time and scalability vs baselines, especially at large scale and high missing ratios, plus complexity analysis of ATPM.
- **[Time Series Continuous Modeling for Imputation and Forecasting with Implicit Neural Representations (TimeFlow)]** _(minor)_ Reviewer T9Ej: add a limitations section noting TimeFlow is one to two orders of magnitude slower at inference than the baselines.

## theory soundness (2)

- **[MissDiff: Training Diffusion Models on Tabular Data with Missing Values]** _(major)_ Theorem 3.2 is questioned: the two arg-min objectives (on observed vs complete data) are not shown to share a solution, and the proof references x(t) which is unavailable at training, making the derivation appear meaningless.
- **[DiffPuter: Empowering Diffusion Models for Missing Data Imputation]** _(major)_ Theorem 1 assumptions (e.g. required convergence type) are not explicitly stated, the proof approximations are unclear, and Remark 2's claim that optimizing the loss implies optimizing likelihood is challenged; sigma(t) is unspecified.

## sharpness (2)

- **[Relational Conformal Prediction for Correlated Time Series (CoREL)]** _(major)_ Improvements on the headline metrics are often only one-to-two percent or in the third/fourth decimal place; a reviewer questioned whether such small interval-width / Winkler-score gains make any practical difference to a decision-maker.
- **[Copula Conformal Prediction for Multi-step Time Series Forecasting (CopulaCPTS)]** _(major)_ Method's main empirical selling point is narrower / more efficient prediction regions than baselines; reviewers wanted this efficiency advantage demonstrated more convincingly, including against additional multi-output CP baselines (e.g. Feldman-Bates-Romano calibrated multi-output quantile regression, Messoudi 2022 ellipsoidal CP).

## more datasets / scale (2)

- **[ProbTS: A Unified Toolkit to Probe Deep Time-series Forecasting]** _(major)_ Benchmarks are mostly small-scale; reviewers repeatedly asked whether the empirical conclusions hold at larger data scale (probabilistic methods may only excel with abundant data) and requested instrumental datasets that were missing (M4, M5, TOURISM, FRED, Monash archive).
- **[ProbTS: A Unified Toolkit to Probe Deep Time-series Forecasting]** _(major)_ Several reviewers said the benchmark datasets are all small-scale and miss field-defining sets (M4, M5, TOURISM, FRED); they argued findings (e.g. that probabilistic methods excel) may not hold at larger scale and that large-scale data is needed to study model scaling and distribution modeling.

## protocol fairness (2)

- **[GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation]** _(major)_ Reviewers said lookback-length tuning was only applied to MOIRAI, not to the other supervised and foundation baselines, even though lookback strongly affects results — making the comparison unfair; also asked for clarification of train/val/test splitting (why 'final window as validation' rather than a standard 7:1:2/6:2:2 ratio) and of zero-shot vs full-shot settings.
- **[Benchmarking Missing Data Imputation Methods in Socioeconomic Surveys]** _(major)_ Reviewers flagged that many baselines used default hyperparameters, which may bias rankings; they asked which methods are most hyperparameter-sensitive, why deep models were not tuned, and whether rankings would shift under tuning.

## statistical tests (2)

- **[Benchmarking Missing Data Imputation Methods in Socioeconomic Surveys]** _(major)_ A reviewer required statistical-significance testing of performance differences between imputation methods, with reported p-values clarifying which method comparisons (especially among top performers) are significant rather than noise.
- **[There are no Champions in Long-Term Time Series Forecasting]** _(major)_ A reviewer pushed back that the paper does not give convincing technical evidence for the recommended statistical tests to determine model superiority, and demanded a deeper technical explanation of why popular metrics fail to robustly identify the best model.

## enthusiasm (1)

- **[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation]** _(minor)_ All three reviewers scored 6/10 ('lack of enthusiasm'); PCs accepted anyway citing the important problem and 40-65% empirical gains.

## no theory / identifiability (1)

- **[Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural Networks (GRIN)]** _(major)_ Reviewer hJvE: insufficient theoretical justification for why the graph representation should yield better performance; add motivation for the inductive bias and ablations (fully-connected vs identity adjacency).

## evaluation protocol (1)

- **[Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations (SPIN)]** _(major)_ Reviewer fr5V: questioned the evaluation methodology, preferring ground-truth reconstruction error over an autoregressive proxy that may be misleading; felt author responses were too brief.

## unsupported claim (1)

- **[Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations (SPIN)]** _(minor)_ Reviewer ajb9: the claim that 'GRIN suffers from error propagation typical of autoregressive models' is unjustified and needs a supporting reference.

## hyperparameter sensitivity (1)

- **[TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis]** _(minor)_ Reviewer v92e: how carefully must the period-count parameter k be chosen, and does larger k help or hurt? Sensitivity unclear.

## information loss (1)

- **[TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis]** _(minor)_ Reviewer gBbZ: the FFT truncate operation is unconvincing as it could blindly discard useful information.

## baseline discrepancy (1)

- **[TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis]** _(minor)_ Reviewer gBbZ: DLinear baseline numbers are inconsistent with its original paper and need elaboration.

## anonymity / leakage (1)

- **[TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis]** _(minor)_ Reviewer v92e & public comment: ethics concern about a near-identical arXiv preprint with author names visible (possible double-blind violation).

## missing methods (1)

- **[TSI-Bench: Benchmarking Time Series Imputation]** _(major)_ Reviewers mYVZ & uzuS: missing recent probabilistic/generative imputers (TimeCIB, GRIN, DSPD-GP, TIDER, BiTGraph, mTANs); models considered are outdated.

## unnatural adaptation (1)

- **[TSI-Bench: Benchmarking Time Series Imputation]** _(major)_ Reviewer mYVZ: adapting forecasting methods to imputation by merely swapping the output layer is 'unnatural' and doesn't reflect their core purpose.

## reproducibility / HP disclosure (1)

- **[TSI-Bench: Benchmarking Time Series Imputation]** _(major)_ Reviewer uzuS: authors don't provide per-method hyperparameters/settings (GP kernels, diffusion noise schedules, timestep configs); unclear if modern methods were tuned optimally on new datasets.

## baseline anomaly (1)

- **[TSI-Bench: Benchmarking Time Series Imputation]** _(major)_ Reviewer uzuS: BRITS outperforms modern methods — is this due to suboptimal tuning of the newer models? Needs explanation.

## static covariates (1)

- **[TSI-Bench: Benchmarking Time Series Imputation]** _(minor)_ Reviewer cU7Z: benchmark gives no support for static info (spatial coordinates, descriptive attributes).

## usability / reproducibility (1)

- **[TSI-Bench: Benchmarking Time Series Imputation]** _(major)_ Reviewer cU7Z: platform not user-friendly — no documentation, hard-coded missing rates and algorithm list, no clear input/output spec, hard to add new datasets; only good for replicating existing experiments.

## anonymity (1)

- **[TSI-Bench: Benchmarking Time Series Imputation]** _(minor)_ Reviewer cU7Z: submission not fully anonymized — an author name appears in supplementary material.

## weak / inconsistent baselines (1)

- **[Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models (SSSD)]** _(major)_ Reviewer 5Tb4: baseline selection lacks motivation; strong baselines (Autoformer, NAOMI, NRTSI, SAITS) appear inconsistently across experiments — apply the strongest baselines consistently everywhere.

## MNAR / missingness terminology (1)

- **[Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models (SSSD)]** _(major)_ Reviewer 5Tb4: missingness terminology is confusing; clarify how RM/MNR/BM/TF map to the standard MCAR/MAR/MNAR taxonomy.

## calibration / robustness (1)

- **[Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models (SSSD)]** _(minor)_ Reviewer 5Tb4: visualizations show random samples rather than distribution means, and large error bars in Table 4 raise robustness questions.

## generalization across missingness (1)

- **[Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models (SSSD)]** _(major)_ Reviewer kCUh: lacks cross-missingness generalization tests (e.g., train at 20% RM, test at 30%; train/inference condition mismatch).

## presentation (1)

- **[HDDI: A Historical Data-Based Diffusion Imputation Method for High Missing Rate and Long-Term Gap Multivariate Time Series]** _(major)_ Core result tables and even symbol definitions are relegated to the appendix while the main text is sparse on experiments; reviewers cited the ICLR guideline that key evidence must be in the main body; results shown only in normalized form, obscuring real-world benefit.

## weak baselines / state of the art (1)

- **[Low Rank Matrix Completion via Robust Alternating Minimization in Nearly Linear Time]** _(major)_ The headline running-time result must be judged against the recent Kelner et al. (2023) algorithm, which achieves the same O(|Omega|k) time but with significantly lower sample complexity (no dependence on the condition number kappa). Reviewer asks the authors to justify the significance of their result given a competing method dominates on sample complexity.

## no empirical validation (1)

- **[Low Rank Matrix Completion via Robust Alternating Minimization in Nearly Linear Time]** _(major)_ The result is purely theoretical and no experiments are provided to demonstrate that the claimed efficiency improvement actually materializes in practice.

## no lower bound / significance framing (1)

- **[Low Rank Matrix Completion via Robust Alternating Minimization in Nearly Linear Time]** _(major)_ Without any discussion of computational limitations or runtime/sample-complexity lower bounds, it is hard to judge whether the near-linear-in-verification-time result is actually significant; the paper should state what runtime one should be aiming for.

## reproducibility / proof rigor (1)

- **[Low Rank Matrix Completion via Robust Alternating Minimization in Nearly Linear Time]** _(major)_ Reviewer is unsure of key steps in the forward-error and matrix-perturbation proofs (e.g., why a Pythagorean identity holds, full-column-rank assumptions, a sigma_min bound), flagging confusing/undefined notation and recommending borderline reject pending clarification.

## presentation / altitude (1)

- **[Low Rank Matrix Completion via Robust Alternating Minimization in Nearly Linear Time]** _(minor)_ Paper spends too much space on intuition and too little on technical detail in the main body; main theorem appears too late, making the paper dry and hard to verify without reading the appendix.

## ablation / design choices (1)

- **[Low Rank Matrix Completion via Robust Alternating Minimization in Nearly Linear Time]** _(minor)_ Reviewer questions whether the per-iteration sample partitioning (sample splitting) can be avoided and asks the authors to clarify whether reusing all samples across iterations breaks correctness or merely complicates the proof.

## inconsistent wins (1)

- **[Multi-Mode Deep Matrix and Tensor Factorization]** _(major)_ The proposed models do not consistently outperform existing solutions on the matrix/tensor completion tasks, weakening the empirical case.

## narrow evaluation / downstream utility (1)

- **[Multi-Mode Deep Matrix and Tensor Factorization]** _(major)_ Method is evaluated only on matrix and tensor completion, raising doubts about usefulness on other downstream tasks; reviewer wants analysis of the recovered latent factors.

## assumption realism (1)

- **[Multi-Mode Deep Matrix and Tensor Factorization]** _(minor)_ Reviewer questions the realism of the generative assumption X = U^T Q V and asks for real-world motivating examples; another reviewer asks that synthetic data be generated under the exact model assumption (matching activation) to verify recovery.

## complexity analysis (1)

- **[Multi-Mode Deep Matrix and Tensor Factorization]** _(minor)_ Computational complexity analysis of the proposed models is not done carefully/elegantly; optimization procedure is not very clearly described.

## assumption realism / structure not universal (1)

- **[TIDER: Multivariate Time-series Imputation with Disentangled Temporal Representations]** _(major)_ The hard-coded decomposition into trend, seasonality, and local/bias factors may not hold across all datasets; some data have no trend or seasonality or contain very unpredictable patterns, and the fixed factors cannot be learned/adapted to unknown or more complex hidden temporal structure.

## hyperparameter sensitivity / tuning (1)

- **[TIDER: Multivariate Time-series Imputation with Disentangled Temporal Representations]** _(major)_ Biggest concern is that the objective involves many hyperparameters; it is unclear how those choices can be made in practice and how sensitive the results are to them — wrong choices could wipe out the reported gains. Reviewer would reconsider rating if hyperparameter guidance and sensitivity were shown.

## weak/insignificant improvements (1)

- **[TIDER: Multivariate Time-series Imputation with Disentangled Temporal Representations]** _(major)_ The method only slightly outperforms baselines on most datasets; it is hard to say the accuracy differences are statistically significant, making results not fully convincing.

## generalization across series (1)

- **[TIDER: Multivariate Time-series Imputation with Disentangled Temporal Representations]** _(minor)_ Training is fit per individual time series; the latent space is not regularized across the dataset to guarantee generalizable disentanglement, which may limit high-dimensional or strongly cross-correlated settings.

## overclaiming / clarity (1)

- **[TIDER: Multivariate Time-series Imputation with Disentangled Temporal Representations]** _(minor)_ Claim that the method 'scales well to large datasets where deep methods struggle' is misleading — it really concerns scaling to long time series, not large datasets; reviewer also flags redundant repetition in the writing and wants a clearer explanation of how factorization actually imputes missing values.

## limited novelty / positioning vs prior work (1)

- **[Online Low Rank Matrix Completion]** _(major)_ The online low-rank completion setting can be viewed as a contextual low-rank matrix bandit; multiple prior works (Sen 2017, Lu 2018, Jun 2019, Huang 2021, Zhou 2020) are not compared, and the claimed improvement (T^{1/2} vs T^{2/3}) only holds for the rank-1 special case while the general algorithm matches existing rates.

## proof rigor / correctness (1)

- **[Online Low Rank Matrix Completion]** _(major)_ Across multiple review rounds the proofs remained 'sloppy' and the manuscript switched back and forth between formulations of the algorithm; a key corollary proof was still incorrect and the resampling procedure's correctness was doubted. Reviewer conditioned a score raise on a clean, self-consistent revision of proofs and appendix.

## calibration / uncertainty metric (1)

- **[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation]** _(major)_ Negative log-likelihood is a standard metric for evaluating uncertainty but is omitted; reviewer asks why such a standard probabilistic-evaluation metric was left out of the probabilistic interpolation/imputation evaluation.

## irregular sampling / preprocessing concern (1)

- **[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation]** _(minor)_ Discretizing irregularly sampled series (e.g., PhysioNet) loses information; reviewer asks how interpolation vs imputation tasks differ after discretization and whether the method can handle irregular sampling directly.

## baselines (1)

- **[Simplified Mamba with Disentangled Dependency Encoding for Long-Term Time Series Forecasting (SAMBA)]** _(major)_ Multiple reviewers demanded stronger/more current baselines: LLM-based methods (LLM4TS, CATS, AutoTimes), dimension-decoupling methods (TimeDRL), and a Patch+MLP comparison that was missing from key tables.

## theory/ablation (1)

- **[Simplified Mamba with Disentangled Dependency Encoding for Long-Term Time Series Forecasting (SAMBA)]** _(minor)_ Reviewers wanted theoretical analysis of how removing activations mitigates overfitting, a direct SAMBA-vs-standard-Mamba block comparison, and noted removing nonlinearities actually degraded performance on Traffic and ETTm1.

## baseline fidelity / leakage (1)

- **[MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting]** _(major)_ A reviewer reimplemented PatchTST and got better numbers than MambaTS on ETTm2 (input length 720), arguing the paper's reported PatchTST baseline numbers were not faithful; the AC cited unresolved concerns 'around fidelity of the reported baseline metrics, especially when baseline models are not trained to convergence.'

## training protocol (1)

- **[MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting]** _(major)_ Training models for only 10 epochs was called unfair; reviewers insisted all models reach convergence (10 vs 100 epochs contested) for a valid comparison.

## scalability / efficiency claims (1)

- **[MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting]** _(major)_ Efficiency was compared only against iTransformer; reviewers demanded comparison with efficient models (DLinear, FITS) and real runtime evidence for the claimed linear complexity, noting the K-1 causal-estimation iterations make the efficiency claim questionable.

## weak baselines / more baselines (1)

- **[MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting]** _(minor)_ Only ~7 baselines; reviewers requested newer post-iTransformer models (ModernTCN, UniTST, TSLANet) and recent SOTA (OneFitsAll/GPT4TS, TimeLLM).

## no identifiability / theory rigor (1)

- **[MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting]** _(major)_ Proposition 2's proof 'does not make sense' / is a procedure description not a proof; the random-walk causal estimation lacked convergence guarantees or confidence intervals, and robustness depends heavily on accuracy of the inferred causal graph.

## long-range underperformance (1)

- **[S7: Selective and Simplified State Space Layers for Sequence Modeling]** _(major)_ On the Long Range Arena benchmark S7 was only marginally better than S5 on a couple of tasks and significantly worse on the rest (and much worse on Path-X), undermining the claim that input-dependent dynamics improve long-range modeling; reviewers wanted this explained.

## efficiency claims (1)

- **[S7: Selective and Simplified State Space Layers for Sequence Modeling]** _(major)_ The paper claims S7 is 'simpler/more efficient' than Mamba but gives no efficiency data (FLOPs, parameter count, runtime); reviewers asked in what concrete sense it is simpler and how it runs efficiently without selective-scan parallelism.

## ablation / robustness (1)

- **[S7: Selective and Simplified State Space Layers for Sequence Modeling]** _(minor)_ Reviewers wanted experiments with different state-matrix initializations (deep SSMs are known to be init-sensitive) and explicit verification of the stable-reparameterization benefit for long-term dependencies.

## no theory / justification (1)

- **[Is Bidirectionality Necessary in Mamba for Time Series Forecasting? (FSMamba)]** _(major)_ Reviewers and AC said the core claim is weak because the paper never formally analyzes why minimizing embedding distance between reversed channel orders should substitute for bidirectionality or yield channel-order invariance; a formal connection between the regularizer and bidirectional behavior was demanded.

## robustness scope (1)

- **[Is Bidirectionality Necessary in Mamba for Time Series Forecasting? (FSMamba)]** _(major)_ The Flipped-Siamese regularizer enforces robustness only to a single permutation (the reverse flip), not general permutation invariance; the paper dismissed random permutations as 'unstable' without deep investigation, so the claimed order-robustness is limited.

## sensitivity / ablation (1)

- **[Is Bidirectionality Necessary in Mamba for Time Series Forecasting? (FSMamba)]** _(minor)_ Lambda sensitivity was reported only on ETTh1 ([0.01,0.1]); reviewers wanted systematic lambda-sensitivity across Weather, ECL, PEMS since the regularizer is central to the method.

## more datasets / scalability (1)

- **[Is Bidirectionality Necessary in Mamba for Time Series Forecasting? (FSMamba)]** _(major)_ Reviewers argued existing benchmarks have too few channels and asked for high-dimensional datasets (Time-HD benchmark, Wike2000 from TFB) to truly validate channel-dependency modeling; authors added M5 (3049 channels) and S&P500 (1475 channels) in rebuttal.

## generality (1)

- **[Is Bidirectionality Necessary in Mamba for Time Series Forecasting? (FSMamba)]** _(minor)_ Reviewers asked whether the two loss strategies transfer to other channel-dependency models (iTransformer, DUET) and whether preserving only linear correlations (vs non-linear) in the CSM pretraining suffices.

## more baselines / horizons (1)

- **[S4M: S4 for Multivariate Time Series Forecasting with Missing Values]** _(minor)_ Reviewers asked how complete-data baselines (Transformer, Autoformer) are adapted to partial data, requested SOTA long-term forecasters (iTransformer, CARD, Crossformer), and varied-horizon experiments with a fixed lookback.

## weak datasets (1)

- **[S4M: S4 for Multivariate Time Series Forecasting with Missing Values]** _(minor)_ Chosen datasets (Traffic, Electricity, ETTh1, Weather) were called fairly simple; reviewers wanted a more comprehensive evaluation across more datasets, and a clearer discussion versus existing missing-value methods.

## hyperparameter guidance (1)

- **[S4M: S4 for Multivariate Time Series Forecasting with Missing Values]** _(minor)_ Key hyperparameters (K1,K2,tau1,tau2) were set empirically; reviewers asked for guidance on choosing them across datasets with different characteristics.

## MNAR / real missingness (1)

- **[DiffImp: Efficient Diffusion Model for Probabilistic Time Series Imputation with Bidirectional Mamba Backbone]** _(major)_ Only synthetic injected missingness was used; reviewers wanted real-world incomplete datasets, and standard imputation benchmarks (PhysioNet 2012 healthcare, AQI air quality, ECG as in SSSD/CSDI) to demonstrate effectiveness.

## calibration / probabilistic claim (1)

- **[DiffImp: Efficient Diffusion Model for Probabilistic Time Series Imputation with Bidirectional Mamba Backbone]** _(minor)_ Reviewers said the probabilistic-imputation claim was not substantiated: the conditioning of the conditional diffusion model and why the model yields calibrated probabilistic imputations were not clearly explained.

## experimental consistency (1)

- **[DiffImp: Efficient Diffusion Model for Probabilistic Time Series Imputation with Bidirectional Mamba Backbone]** _(minor)_ Inconsistent missing-rate grids across datasets (70/80/90% on MuJoCo vs 10/30/50% on Electricity) were questioned, with a request to use a consistent set of ratios.

## theory (1)

- **[DiffImp: Efficient Diffusion Model for Probabilistic Time Series Imputation with Bidirectional Mamba Backbone]** _(minor)_ A reviewer noted there is no theoretical result supporting the proposed techniques.

## theory-practice gap (1)

- **[MissDiff: Training Diffusion Models on Tabular Data with Missing Values]** _(major)_ The network takes the observed coordinates x^obs as input, not the full x; the claim that masked/marginal score matching recovers the complete-data (joint) score is unsubstantiated for the actual network input.

## recurring-demand-synthesis (1)

- **[Missing Data Imputation by Reducing Mutual Information with Rectified Flows (and adjacent diffusion-tabular imputation submissions)]** _(major)_ Across these tabular/deep-generative imputation threads the recurring reviewer demands that a method/benchmark paper like CAFE would also face are: (1) prove baselines were tuned fairly (multiple papers were dinged or rejected because baselines underperformed mean imputation); (2) report runtime/compute/scalability on large high-dimensional data, not just small UCI sets; (3) evaluate explicitly under MCAR/MAR/MNAR with stated mechanism implementations and identifiability discussion; (4) show downstream-task utility, not only RMSE/MAE; (5) differentiate novelty crisply from near-neighbour methods; (6) handle/evaluate mixed numeric+categorical features; and (7) make theoretical claims airtight (assumptions stated, no train-time use of unavailable quantities, train/test mask-distribution match).

## no identifiability (1)

- **[Relational Conformal Prediction for Correlated Time Series (CoREL)]** _(major)_ The validity proof (Prop 3.1) hides everything that could break the conformal guarantee inside a total-variation term that is impossible to estimate or control in practice; the asymptotic-shrinkage claim (under stationarity / strong mixing) is asserted with no proof or supporting citation, unlike CQR/SPCI which give an actual finite validity proof.

## large calibration set (1)

- **[Copula Conformal Prediction for Multi-step Time Series Forecasting (CopulaCPTS)]** _(major)_ The two-stage calibration splits the calibration data a second time, so the method needs a very large calibration set — a luxury rarely available in practice; this large-data requirement was flagged as a major weakness limiting practical significance.

## exchangeability vs temporal dependence (1)

- **[Copula Conformal Prediction for Multi-step Time Series Forecasting (CopulaCPTS)]** _(major)_ Reviewers argued the 'time series' framing is overstated: the theory imposes i.i.d./exchangeability across the series z^i, eliminating temporal dependence, so within each step the data points are exchangeable and ordinary CP applies; they asked what this assumption really imposes and how it differs from work that goes beyond exchangeability.

## theory / definitions (1)

- **[Copula Conformal Prediction for Multi-step Time Series Forecasting (CopulaCPTS)]** _(minor)_ Reviewers questioned the specific multivariate empirical quantile definition (Eq.13) — is it the only choice, why this one — the choice of L1/L2 norm in the nonconformity score and its effect, a marginal-vs-joint inconsistency in the validity statement (Eq.4 vs the proof), and flagged typos in the loss and proof equations needing double-checking.

## leakage / fairness (1)

- **[Copula Conformal Prediction for Multi-step Time Series Forecasting (CopulaCPTS)]** _(minor)_ A reviewer asked whether calibration-free baselines were given the extra data that the proposed method spends on its calibration split, since otherwise the comparison is unfair to methods that do not need a calibration set.

## coverage guarantee (1)

- **[Conformal PID Control for Time Series Prediction]** _(major)_ Multiple reviewers pressed that the coverage guarantee is only asymptotic (longitudinal/long-run coverage) with no finite-sample result and no explicit lower/upper bounds; they questioned the value of an asymptotic-only guarantee given that CP's appeal is finite-sample validity, and asked whether asymptotic coverage is provably the best achievable in the adversarial setting.

## hyperparameters (1)

- **[Conformal PID Control for Time Series Prediction]** _(major)_ The framework has many non-intuitive design choices and hyperparameters (saturation function constants C_sat and K_I, learning rate eta, the score-tracker r_t); reviewers said the appendix figures show the choices matter enormously, the paper implies trial-and-error tuning, and asked for principled guidance on selecting them.

## delayed feedback scope (1)

- **[Conformal PID Control for Time Series Prediction]** _(minor)_ Reviewer noted the method assumes the response y_t is revealed at every step and asked about settings where feedback arrives delayed or in batches.

## evaluation realism (1)

- **[Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting (TSDiff)]** _(minor)_ The 'prediction refinement' evaluation (refining a weak forecaster) was called artificial / lacking a real application scenario — if you already have the diffusion model, why not forecast with it directly; reviewers wanted the evaluation focused on more practical settings.

## theory / derivation rigor (1)

- **[Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting (TSDiff)]** _(major)_ Reviewers wanted proofs that the self-guidance objective reduces to the maximum-likelihood / unconditional p(y) in the appropriate limits, asked how numerical singularities in the guidance term are handled, and argued the Bayesian framing (unit-variance Gaussian prior giving L2, plus the quantile regularizer) is awkward and essentially just optimization-based regularization dressed up as Bayes.

## calibration / metrics (1)

- **[ProbTS: A Unified Toolkit to Probe Deep Time-series Forecasting]** _(minor)_ Reviewers found the CRPS-vs-NMAE analysis (probabilistic forecasting models good at distribution modeling but producing poor point forecasts) the most valuable insight, but wanted it developed more rigorously; one also flagged CRPS being used before being defined.

## MNAR / multivariate (1)

- **[ProbTS: A Unified Toolkit to Probe Deep Time-series Forecasting]** _(major)_ A reviewer noted the conspicuous absence of the multivariate / cross-channel dimension — channel-independent methods like PatchTST ignore cross-channel interactions — and asked whether and how that limits the compared methods.

## benchmark positioning / novelty (1)

- **[ProbTS: A Unified Toolkit to Probe Deep Time-series Forecasting]** _(major)_ As a library/benchmark paper, reviewers demanded a proper comparison to existing toolkits (GluonTS, TSLib, Monash) rather than to individual models, more library attributes compared (metrics, datasets, transforms, data-loading, out-of-RAM handling), and contested the dismissal of GluonTS; the rejecting reviewer said it 'tries to do too much and too little', lacking implementation detail, comprehensive coverage, and definitive insights.

## reproducibility / leakage (1)

- **[ProbTS: A Unified Toolkit to Probe Deep Time-series Forecasting]** _(minor)_ Reviewers asked how hyperparameters and preprocessing were chosen per dataset (these dramatically affect results and were not disclosed), whether standard tuning (e.g. raytune) was used, how patch sizes were set, and whether reproduced numbers match each method's originally reported numbers.

## insights / analysis depth (1)

- **[ProbTS: A Unified Toolkit to Probe Deep Time-series Forecasting]** _(major)_ Several reviewers felt the empirical insights were under-developed: the paper does not give a quantifiable rule for which approach (architecture-centric vs probabilistic, autoregressive vs non-autoregressive, short- vs long-horizon) suits which dataset, and suggested synthetic/controlled experiments to disentangle confounded data characteristics.

## fair comparison (1)

- **[TSI-Bench: Benchmarking Time Series Imputation]** _(major)_ Reviewer uzuS asked why an old method (BRITS) outperforms modern methods — is this inadequate optimization/tuning of the newer baselines rather than a real result? Implies the benchmark may be unfair to some methods.

## novelty/insight depth (1)

- **[TSI-Bench: Benchmarking Time Series Imputation]** _(major)_ Reviewers (Evcs, mYVZ) said the paper reads as a performance comparison without analytical depth — it does not explain when or why particular methods excel; findings (missing patterns hurt performance, forecasting architectures adapt to imputation, imputation helps downstream) are unsurprising. Better suited to a journal survey.

## adaptation validity (1)

- **[TSI-Bench: Benchmarking Time Series Imputation]** _(minor)_ Reviewers questioned the 'tailoring forecasting algorithms for imputation' contribution as artificial — simply swapping output layers does not reflect what forecasting models are designed to do.

## reproducibility/usability (1)

- **[TSI-Bench: Benchmarking Time Series Imputation]** _(major)_ Reviewer cU7Z found the platform not usable by external researchers: no documentation, hardcoded settings/algorithms requiring source edits, unclear I/O, hard to integrate new datasets; also flagged an anonymity violation (author name in code).

## metrics/statistical rigor (1)

- **[GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation]** _(major)_ Multiple reviewers attacked the use of MAPE as the sole point-forecast metric (it favors underprediction, is unstable near zero) and demanded additional robust metrics (sMAPE, ND, MASE, MSE/MAE) and geometric-mean aggregation of normalized metrics to avoid misleading rankings.

## more datasets / redundancy (1)

- **[GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation]** _(major)_ Reviewers questioned the rationale for the dataset/frequency selection: the set heavily overlaps MOIRAI's collection, may be redundant and energy-inefficient, and the authors should show each added dataset/frequency covers a distinctive pattern (not already in existing benchmarks) and compute the effective rank of the dataset-by-model result matrix.

## validation against prior work (1)

- **[GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation]** _(minor)_ A reviewer asked the authors to cross-check results on shared classical datasets (ETTh, Electricity, Solar, M4) against existing published numbers to demonstrate the protocol is reliable and models are correctly reproduced, and to justify excluding common datasets (Traffic, Wikipedia, Exchange).

## ablation/analysis depth (1)

- **[GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation]** _(major)_ Reviewers said the study only reports aggregate wins per dataset without consistent, generalizable conclusions; demanded results sliced by the six time-series features (trend, seasonality, entropy, Hurst, stability, lumpiness) and analysis of the pretraining data, plus explanation of counterintuitive findings (e.g. why multivariate-aware MOIRAI does best on univariate data).

## baseline against existing toolkits (1)

- **[ProbTS: A Unified Toolkit to Probe Deep Time-series Forecasting]** _(major)_ A reviewer said Table 1 conflates a library with models and demanded comparison against existing libraries (GluonTS, TSLib, Darts, Merlion, Kats) on concrete attributes (metrics, datasets, transforms, data-loading efficiency), and disputed the claim that GluonTS only supports a single paradigm.

## insight depth (1)

- **[ProbTS: A Unified Toolkit to Probe Deep Time-series Forecasting]** _(major)_ Reviewers said the empirical study yields no definitive, quantifiable guidance on the interplay between architecture design and probabilistic forecasting head, evaluates only a small subset of architecture x head combinations, and would benefit from synthetic/controlled experiments to isolate confounds among datasets with diverse characteristics.

## multivariate (1)

- **[ProbTS: A Unified Toolkit to Probe Deep Time-series Forecasting]** _(minor)_ A reviewer noted the absence of treatment of the multivariate/cross-channel nature of time series (e.g. channel-independent PatchTST) and asked whether ignoring cross-channel interactions limits some methods.

## temporal structure (1)

- **[Benchmarking Missing Data Imputation Methods in Socioeconomic Surveys]** _(major)_ A reviewer noted longitudinality is a stated motivation and asked whether the missingness generation preserves time dependencies and whether the evaluation actually reflects temporal structure.

## actionability (1)

- **[Benchmarking Missing Data Imputation Methods in Socioeconomic Surveys]** _(minor)_ A reviewer asked for a 'Practical Recommendations' section and a failure-modes/diagnostic analysis so applied users can act on the findings (e.g. MissForest/HyperImpute strong at low missingness, deep models at high missingness).

## protocol fragility (1)

- **[There are no Champions in Long-Term Time Series Forecasting]** _(minor)_ A reviewer endorsed the core finding that benchmark rankings are fragile to hyperparameters and evaluation pipelines (and that by cherry-picking a metric/story any method can be crowned 'champion'), but said current benchmarks still miss domains where traditional methods may beat DL and lack relevant context features (news for finance, geometry for weather).

## actionable guidance (1)

- **[There are no Champions in Long-Term Time Series Forecasting]** _(major)_ A reviewer said the paper diagnoses problems (weak metrics, non-representative datasets, complex models not beating DLinear) but fails to give concrete prescriptions: how to build more robust datasets, exactly how metrics fail technically and how to use them, and what fundamentally limits complex architectures.

## more datasets / novelty of benchmark (1)

- **[GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation]** _(major)_ Reviewer FpFr: the datasets heavily overlap with MOIRAI's existing collection; the benchmark is mainly a quantity increase (more datasets/frequencies) and it is weakly argued why more evaluation datasets are needed or what unique insights they reveal that prior work did not; covering an excess of datasets also burdens resource-limited groups.

## metrics / leaderboard (1)

- **[GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation]** _(major)_ Reviewer ah9K: MAPE is a poor point-forecast metric (favors underprediction, blows up near zero); switch to or also report more robust metrics like MASE, and use geometric means when aggregating normalized metrics.

## slicing / stratified analysis (1)

- **[GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation]** _(major)_ Reviewers ah9K & Pz22: given the dataset properties (trend, seasonality, entropy, Hurst, stability, lumpiness), report benchmark results sliced/stratified on these properties rather than only headline numbers; questioned whether splitting by number-of-variates (multivariate vs univariate) is meaningful since not all multivariate sets have strong cross-channel correlation, so claims that a model is best on 'multivariate' data may mislead.

## leakage / pretraining data (1)

- **[GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation]** _(major)_ Reviewers SxEF & Pz22 & Pz22: the value and usage of the supplied 'non-leaking' pretraining set is unclear (are all foundation models re-trained on it vs their public checkpoints? is it better than their original corpora?); only test-data features are analyzed -- the pretraining data must also be characterized to tell whether good performance is from pretraining-distribution overlap vs true generalization.

## experimental clarity / protocol (1)

- **[GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation]** _(major)_ Reviewer Pz22: experimental settings are unclear -- input/output lengths for short/medium/long horizons, whether foundation models are evaluated zero-shot or full-shot, and the train/val/test split construction ('final window of training serves as validation', why not a standard 7:1:2 / 6:2:2 ratio) all need to be spelled out.

## covariates handling (1)

- **[GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation]** _(minor)_ Reviewer SxEF: the paper mentions covariates but it is unclear how covariates are incorporated and whether all baselines can forecast with covariates.

## leakage / domain alignment (1)

- **[This Time is Different: An Observability Perspective on Time Series Foundation Models (Toto + BOOM)]** _(major)_ Reviewer pfs4: although the authors claim no leakage between Toto's pretraining corpus and BOOM via environment isolation, both come from the same commercial observability platform, so the performance edge may reflect domain/distribution alignment rather than architectural superiority; framing it as a general-purpose foundation model overgeneralizes since model and benchmark are both observability-tailored.

## weak / unfair baselines (1)

- **[This Time is Different: An Observability Perspective on Time Series Foundation Models (Toto + BOOM)]** _(major)_ Reviewer RNqe: comparing only against zero-shot foundation models is unfair because Toto used far more observability pretraining data; should also evaluate smaller supervised models, and add missing foundation-model baselines (VisionTS, Time-MoE) in Table 3.

## scalability / efficiency (1)

- **[This Time is Different: An Observability Perspective on Time Series Foundation Models (Toto + BOOM)]** _(minor)_ Reviewer WAXQ: a comprehensive efficiency comparison against existing foundation models (MOIRAI, Time-MoE) is missing.

## benchmark label quality / reproducibility (1)

- **[This Time is Different: An Observability Perspective on Time Series Foundation Models (Toto + BOOM)]** _(minor)_ Reviewer cYnS: clarify the BOOM data split (point-level vs query-level, how heterogeneity is handled) and the LLM-plus-human labeling pipeline's quality control / inter-annotator agreement.

## fine-tuning (1)

- **[This Time is Different: An Observability Perspective on Time Series Foundation Models (Toto + BOOM)]** _(minor)_ Reviewer RNqe: main experiments are all zero-shot; it would help to fine-tune Toto and report behavior at different fine-tuning levels.

## method too simple to capture true dependence (1)

- **[Rethinking Channel Dependence for Multivariate Time Series Forecasting: Learning from Leading Indicators (LIFT)]** _(major)_ Reviewers MEf1, cNYF & 4A7m: leading indicators are detected only via pairwise cross-correlation, which cannot capture joint/multi-variable or chain-structured influence (e.g. in X1->X2->...->XC, XC may correlate more with distant variates than with its true driver XC-1), and may mis-specify dependence; cross-correlation is not causation -- can the method be extended to discover Granger/causal relations?

## locally-stationary assumption vs long horizons (1)

- **[Rethinking Channel Dependence for Multivariate Time Series Forecasting: Learning from Leading Indicators (LIFT)]** _(major)_ Reviewer cNYF: the lead-lag relationship is assumed stationary only over short windows, which suggests it should degrade in long-term forecasting, yet the paper claims strong long-horizon performance -- this needs elaboration.

## scalability / fair compute comparison (1)

- **[Rethinking Channel Dependence for Multivariate Time Series Forecasting: Learning from Leading Indicators (LIFT)]** _(minor)_ Reviewer cNYF: as a plug-in added on top of a backbone, a fair comparison of total computational cost/load is needed, otherwise improvement from injected inductive bias is unsurprising.

## contradictory results (1)

- **[Rethinking Channel Dependence for Multivariate Time Series Forecasting: Learning from Leading Indicators (LIFT)]** _(major)_ Reviewer 4A7m: the motivation is to exploit channel dependence, yet channel-independent backbones show larger improvement than channel-dependent ones -- this apparent contradiction needs explanation.

## normalization / clarity (1)

- **[Rethinking Channel Dependence for Multivariate Time Series Forecasting: Learning from Leading Indicators (LIFT)]** _(minor)_ Reviewer JmQw: unclear how instance normalization/denormalization affects results; minor figure-labeling and wording issues.

## no theory / weak justification (1)

- **[Partial Channel Dependence with Channel Masks for Time Series Foundation Models (PCD)]** _(major)_ Reviewers 21yw, BSUW & JKMr: the paper lacks theoretical grounding for why partial channel dependence helps and for the 'implicit heterogeneity' concept; needs a rigorous definition, an argument for why fixed dependence wouldn't work equally well, and analysis of when partial dependence provably beats the full-CD / full-CI extremes.

## missing baseline comparisons (1)

- **[Partial Channel Dependence with Channel Masks for Time Series Foundation Models (PCD)]** _(major)_ Reviewers BSUW, bW9C & saeS: missing comparisons against the same architecture with full channel dependence and full channel independence, against recent MLP/CNN/GNN models (RLinear, ModernTCN, CrossGNN), and against competing plug-ins like LIFT and clustering-based CCM; some reported iTransformer/PatchTST numbers differ markedly from the original papers.

## strong assumption / generality (1)

- **[Partial Channel Dependence with Channel Masks for Time Series Foundation Models (PCD)]** _(major)_ Reviewers BSUW & saeS: the method requires knowing which sample belongs to which dataset at pretraining, a strong assumption when sources are mixed/unknown; and not all foundation models use channel attention (TTM, TimesFM, MOMENT), limiting the plug-in's generality.

## validation on uncorrelated / synthetic data (1)

- **[Partial Channel Dependence with Channel Masks for Time Series Foundation Models (PCD)]** _(minor)_ Reviewer 21yw: should test on synthetic datasets with uncorrelated channels to confirm the channel mask correctly reduces to channel-independence, and verify the CD-ratio metric is consistent across different backbone models.

## scalability / overhead (1)

- **[Partial Channel Dependence with Channel Masks for Time Series Foundation Models (PCD)]** _(minor)_ Reviewer BSUW: quantify computational overhead and memory of the approach relative to full-dependence and full-independence cases.

## baseline tuning / reproducibility (1)

- **[Time Series Continuous Modeling for Imputation and Forecasting with Implicit Neural Representations (TimeFlow)]** _(major)_ Reviewer 3vms: needs much more care describing how baseline hyperparameters were set (recommended defaults vs tuned, convergence/learning curves, architecture-size sweeps); worry that baselines would do substantially better with task-specific tuning.

## gains attributable to inductive bias, not method (1)

- **[Time Series Continuous Modeling for Imputation and Forecasting with Implicit Neural Representations (TimeFlow)]** _(major)_ Reviewer 3vms: the datasets have strong periodicity and the frequency-embedding gives a favorable inductive bias; clarify which baselines share that bias and whether a simpler periodic-kernel GP (or Fourier-feature regression model) would do as well.

## leakage / evaluation honesty (1)

- **[Time Series Continuous Modeling for Imputation and Forecasting with Implicit Neural Representations (TimeFlow)]** _(major)_ Reviewer T9Ej: state prominently and early that the main results are evaluated on the same time series (functions) seen during training; the generalization-to-novel-sequence results are buried in the appendix and show the method's advantage over PatchTST disappears -- these should be surfaced, not downplayed.

## more datasets / scope (1)

- **[Time Series Continuous Modeling for Imputation and Forecasting with Implicit Neural Representations (TimeFlow)]** _(minor)_ Reviewer kiRB: extend evaluation to non-seasonal time series (e.g. weekly-frequency datasets) and add discussion/comparison to forecasting regression models built on Fourier features.

## multivariate / cross-channel handling (1)

- **[TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting]** _(major)_ Reviewers n697 & n697: clarify whether the minimalist low-rank model supports multiple temporal patterns (e.g. daily and weekly periods in traffic) and how it extends to multivariate time series with cross-channel structure.

## CPU vs GPU efficiency claim (1)

- **[TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting]** _(major)_ Reviewer 2K2o: inference time is measured on CPU; questions why, and whether the efficiency gap over baselines would shrink on GPU -- asks for a GPU comparison for a balanced evaluation.

## baseline tuning / protocol (1)

- **[TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting]** _(major)_ Reviewer 2K2o: since experiments use new datasets, was hyperparameter tuning done for the baselines, and what was the search space; also missing a look-back-window-size sensitivity study even though window size strongly affects results.

## missing closely-related work (1)

- **[TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting]** _(major)_ Reviewer 2K2o: FITS and SparseTSF are foundational prior lightweight/low-rank works that underpin this paper and must be discussed with clear distinctions; SparseTSF is missing as a baseline in comparisons.

## underspecified claims / design choices (1)

- **[TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting]** _(minor)_ Reviewer 2K2o: the claim that the model 'captures essential patterns and interactions' does not specify what patterns/interactions; clarify how dataset-level singular values are computed and report the chosen segment length P per dataset.

## motivation / real-world applicability (1)

- **[TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting]** _(minor)_ Reviewers JfQc & Pc2P: deepen the motivation by identifying concrete low-resource real-world deployment scenarios; experiments use only a 720 input length, and the choice of segment length P without prior knowledge needs a principled selection method.

## novelty/contribution (1)

- **[TSI-Bench: Benchmarking Time Series Imputation]** _(major)_ All methods and datasets are taken from prior work; findings (missing pattern hurts performance, forecasting backbones work for imputation) are unsurprising, so it reads as a survey rather than a contribution.

## interpretability (1)

- **[Filling the Gaps: Multivariate Time Series Imputation by Graph Neural Networks (GRIN)]** _(major)_ Insufficient theoretical or intuitive explanation for WHY the graph representation improves performance; wants discussion of inductive bias/regularization effects and ablations over graph structure (fully connected vs identity vs learned).

## task motivation (1)

- **[Filling the Gaps: Multivariate Time Series Imputation by Graph Neural Networks (GRIN)]** _(minor)_ Unclear whether imputation is needed as a standalone task if forecasting methods can consume missing data directly; clarify imputation vs preprocessing framing.

## prior work coverage (1)

- **[Filling the Gaps: Multivariate Time Series Imputation by Graph Neural Networks (GRIN)]** _(major)_ Public comment: substantial prior work from transportation/traffic engineering (e.g. Transdim, kriging, graph signal interpolation) is uncited and uncompared.

## unsupported claims (1)

- **[Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations (SPIN)]** _(major)_ Claims that GRIN/autoregressive models suffer from error propagation are asserted without supporting citations or evidence.

## evaluation methodology (1)

- **[Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations (SPIN)]** _(major)_ Evaluation methodology questioned: reviewer preferred direct RMSE on complete sequences over an autoregressive-based proxy for reconstruction error.

## graph construction (1)

- **[Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations (SPIN)]** _(minor)_ Graph built naively from pairwise distances; physical issues like altitude differences between sensors are not addressed.

## limitations (1)

- **[Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations (SPIN)]** _(minor)_ No explicit limitations discussion.

## analysis depth (1)

- **[TSI-Bench: Benchmarking Time Series Imputation]** _(major)_ Analysis is shallow — reports metrics without explaining WHY particular methods excel in given contexts.

## usability (1)

- **[TSI-Bench: Benchmarking Time Series Imputation]** _(major)_ Platform usability poor: missing documentation, hard-coded parameters, no support for static/spatial attributes, unclear algorithm I/O spec, and difficult dataset/algorithm integration; also an anonymity breach (author name in supplementary code).

## overclaiming (1)

- **[Time Series Continuous Modeling for Imputation and Forecasting with Implicit Neural Representations (TimeFlow)]** _(minor)_ Overstated 'first continuous framework' claims and unsupported assertions about transformer 'significant performance degradation'; asked to back claims with metrics and de-emphasize forecasting in favor of imputation.
