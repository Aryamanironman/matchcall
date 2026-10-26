# Literature Review — Broadcast Football Video to Spoken Commentary

**Project:** football-commentator (Aryaman)  
**Date:** 26 October 2026  
**Scope:** five-stage pipeline — action spotting/event classification → YOLO + ByteTrack tracking and team/possession enrichment → deterministic grounded-description layer → GPT-2 LoRA commentary with a grounding validator → Fish Audio TTS over match audio, with ClearML experiment tracking.

This document is a working literature review: a verified reading list, the themes that structure the field, and the gaps this project addresses. Every citation was checked against a publisher, arXiv, ACL Anthology, CVF, IEEE, ACM, DBLP, or official documentation; non-peer-reviewed sources are labelled.

---

## 0. How the literature maps onto our pipeline

| Stage | What we actually build | Key literature | Our choice / gap |
|---|---|---|---|
| 1. Action spotting + event classification | Score-bug OCR anchors, 2.5 s window scan, YOLO person/ball detector, ResNet-18 event classifier, crowd-audio rise, moment selection | SoccerNet (2018), SoccerNet-v2 (2021), CALF (2020), NetVLAD++ (2021), E2E-Spot (2022), SoccerNet Challenge reports (2023/24) | We use a lightweight, broadcast-specific scan rather than a large spotting network; the score bug gives high-precision goal anchors. |
| 2. Tracking + teams + possession | Aryaman YOLOv5lu / YOLO26 fine-tune, ByteTrack, KMeans jersey colours, nearest-player possession, bounded ball interpolation | YOLO (2016), ByteTrack (2022), SportsMOT (2023), SoccerNet-Tracking (2022), DeepPlayer-Track (2022), Guo et al. (2026) | We combine label-free jersey clustering with a possession heuristic and a bounded ball path that rejects teleports — a deployable alternative to full tracking benchmarks. |
| 3. Grounded description | Deterministic fact sheet: score, minute, ball zone, possession, tracked-player count, event team | MIKE (1998), MatchTime (2024), SoccerNet-Caption (2023), FActScore (2023), RAG (2020) | We separate fact selection from surface realisation; no VLM is allowed to invent facts. This is the audit layer the LLM literature rarely exposes. |
| 4. LLM commentary | GPT-2 + LoRA, completion-only loss, `[HISTORY]/[EVENT]/[INTENSITY]/[COMMENTARY]` markup, grounding validator, template fallback | MatchTime (2024), SoccerNet-Caption (2023), TimeSoccer (2025), SoccerComment (2025), LoRA (2022), FActScore (2023) | Small-GPU LoRA instead of a 7B MLLM; a deterministic validator instead of retrieval-only grounding. |
| 5. TTS + render | Fish Audio s2.1-pro-free, event-conditioned speed, full-length stadium mix, 18 s clips | Tacotron 2 (2017), FastSpeech 2 (2021), VITS (2021), StyleTTS 2 (2023), controllable-TTS survey (2024) | Production TTS with event-based pacing; no peer-reviewed sports-commentary TTS benchmark exists, so we evaluate MOS/RTF ourselves. |
| MLOps | ClearML tasks, scalars, artifacts, model registry for YOLO and GPT-2 | ClearML docs | Every training run is reproducible and auditable. |

---

## 1. Scope and research questions

- **RQ1:** How precisely can broadcast football events be spotted and classified, and what temporal tolerances are meaningful for commentary?
- **RQ2:** Which detection, tracking, team-identification, and possession methods produce reliable player-level facts under broadcast occlusion and on consumer GPUs?
- **RQ3:** How have commentary systems moved from templates to neural/LLM generation, and what data alignment and grounding do they require?
- **RQ4:** How can generated commentary be validated against a deterministic fact layer to control hallucination?
- **RQ5:** What TTS methods provide controllable, low-latency speech, and how should the full pipeline be evaluated and reproduced?

---

## 2. Themes

### 2.1 Football video understanding and action spotting

- **Giancola, Amine, Dghaily & Ghanem (2018), "SoccerNet: A Scalable Dataset for Action Spotting in Soccer Videos," CVPR Workshops, 1711–1721.** [CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w34/html/Giancola_SoccerNet_A_Scalable_CVPR_2018_paper.html) · DOI [10.1109/CVPRW.2018.00223](https://doi.org/10.1109/CVPRW.2018.00223). 500 matches, 764 hours, 6,637 anchors, three event classes; defines action spotting and Average-mAP. Relevance: canonical corpus and metric for event timing.
- **Deliège, Cioppa, Giancola, et al. (2021), "SoccerNet-v2: A Dataset and Benchmarks for Holistic Understanding of Broadcast Soccer Videos," CVPR Workshops, 4508–4519.** [CVF](https://openaccess.thecvf.com/content/CVPR2021W/CVSports/html/Deliege_SoccerNet-v2_A_Dataset_and_Benchmarks_for_Holistic_Understanding_of_Broadcast_CVPRW_2021_paper.html) · DOI [10.1109/CVPRW53098.2021.00508](https://doi.org/10.1109/CVPRW53098.2021.00508). Adds roughly 300k annotations, 17 action classes, camera-shot segmentation, and replay grounding. Relevance: richer event taxonomy for the classification stage.
- **Cioppa, Deliège, Giancola, et al. (2020), "A Context-Aware Loss Function for Action Spotting in Soccer Videos," CVPR, 13126–13136.** [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Cioppa_A_Context-Aware_Loss_Function_for_Action_Spotting_in_Soccer_Videos_CVPR_2020_paper.html) · DOI [10.1109/CVPR42600.2020.01314](https://doi.org/10.1109/CVPR42600.2020.01314). CALF models temporal context around sparse anchors and improves the SoccerNet baseline by 12.8%. Relevance: handles label sparsity and temporal ambiguity in long broadcasts.
- **Giancola & Ghanem (2021), "Temporally-Aware Feature Pooling for Action Spotting in Soccer Broadcasts," CVPR Workshops, 4490–4499.** [CVF](https://openaccess.thecvf.com/content/CVPR2021W/CVSports/html/Giancola_Temporally-Aware_Feature_Pooling_for_Action_Spotting_in_Soccer_Broadcasts_CVPRW_2021_paper.html) · DOI [10.1109/CVPRW53098.2021.00506](https://doi.org/10.1109/CVPRW53098.2021.00506). NetVLAD++ pools past and future context separately and reaches 53.4% Average-mAP. Relevance: efficient, strong feature-aggregation baseline.
- **Hong, Zhang, Gharbi, Fisher & Fatahalian (2022), "Spotting Temporally Precise, Fine-Grained Events in Video," ECCV, 33–51.** [ECVA](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3387_ECCV_2022_paper.php) · DOI [10.1007/978-3-031-19833-5_3](https://doi.org/10.1007/978-3-031-19833-5_3). E2E-Spot is a compact end-to-end model trained on 100-frame clips on a single GPU, evaluated on SoccerNet-v2 among others. Relevance: precise timestamps with modest compute.
- **Cioppa, Giancola, Somers, Magera, Zhou, Mkhallati, et al. (2023), "SoccerNet 2023 Challenges Results," arXiv:2309.06006.** [arXiv](https://arxiv.org/abs/2309.06006); see also the 2024 report, [arXiv:2409.10587](https://arxiv.org/abs/2409.10587). Seven tasks; action-spotting tight Average-mAP reached 71.31; dense captioning improved with LLMs. Relevance: current SOTA and evidence that spotting is maturing while captioning is advancing. ASTRA (Xarles, Escalera, Moeslund & Clapés, MMSports 2023, pp. 93–102, DOI [10.1145/3606038.3616153](https://doi.org/10.1145/3606038.3616153)) placed third with 66.82 tight Average-mAP and used audio for non-visible actions.

### 2.2 Detection, multi-object tracking, team identification and possession

- **Redmon, Divvala, Girshick & Farhadi (2016), "You Only Look Once: Unified, Real-Time Object Detection," CVPR, 779–788.** [CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html) · DOI [10.1109/CVPR.2016.91](https://doi.org/10.1109/CVPR.2016.91). Single-pass regression at 45 FPS (155 FPS for Fast YOLO). Relevance: foundation of the real-time detector family used here.
- **Ultralytics (2023), YOLOv8 documentation.** [Models](https://docs.ultralytics.com/models/yolov8/) · [Tracking mode](https://docs.ultralytics.com/modes/track/). Official software documentation for detection, segmentation, classification, pose, and OBB; tracking mode supports BoT-SORT and ByteTrack. Relevance: practical deployment path for YOLO weights and tracker integration. Non-peer-reviewed software documentation.
- **Zhang, Sun, Jiang, et al. (2022), "ByteTrack: Multi-Object Tracking by Associating Every Detection Box," ECCV.** [arXiv](https://arxiv.org/abs/2110.06864) · DOI [10.1007/978-3-031-20047-2_1](https://doi.org/10.1007/978-3-031-20047-2_1). Associates low-score boxes to recover occluded objects; 80.3 MOTA, 77.3 IDF1, 63.1 HOTA on MOT17 at about 30 FPS. Relevance: robust tracking under broadcast occlusion.
- **Cui, Zeng, Zhao, Yang, Wu & Wang (2023), "SportsMOT: A Large Multi-Object Tracking Dataset in Multiple Sports Scenes," ICCV, 9921–9931.** [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Cui_SportsMOT_A_Large_Multi-Object_Tracking_Dataset_in_Multiple_Sports_Scenes_ICCV_2023_paper.html) · DOI [10.1109/ICCV51070.2023.00910](https://doi.org/10.1109/ICCV51070.2023.00910). 240 sequences, over 150k frames, 1.6M boxes across football, basketball, and volleyball; MixSort baseline. Relevance: sports-specific fast motion and similar-appearance challenges.
- **Cioppa, Giancola, Deliège, et al. (2022), "SoccerNet-Tracking: Multiple Object Tracking Dataset and Benchmark in Soccer Videos," CVPR Workshops, 3491–3502.** [CVF](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/html/Cioppa_SoccerNet-Tracking_Multiple_Object_Tracking_Dataset_and_Benchmark_in_Soccer_Videos_CVPRW_2022_paper.html) · DOI [10.1109/CVPRW56347.2022.00393](https://doi.org/10.1109/CVPRW56347.2022.00393). 200 thirty-second sequences plus a full 45-minute half, with boxes and tracklet IDs; motivates possession and formation analysis. Relevance: closest soccer-specific tracking benchmark.
- **Team ID and possession:** **Naik, Hashmi, Geem & Bokde (2022), "DeepPlayer-Track: Player and Referee Tracking With Jersey Color Recognition in Soccer," IEEE Access 10, 32494–32509.** [IEEE](https://ieeexplore.ieee.org/document/9739737) · DOI [10.1109/ACCESS.2022.3161441](https://doi.org/10.1109/ACCESS.2022.3161441). YOLOv4 plus JCD-SORT with jersey-colour masks; 96% MOTA at 23 FPS. **Guo, Zeng, Deng, et al. (2026), "Automatic Estimation of Football Possession via Improved YOLOv8 Detection and DBSCAN-Based Team Classification," Sensors 26(4), 1252.** DOI [10.3390/s26041252](https://doi.org/10.3390/s26041252). YOLOv8 variants, DBSCAN jersey clustering, and Norfair tracking; RMSE 4.87 on 20 full-match clips. Relevance: label-free team assignment plus possession, directly analogous to the project's fact layer.

### 2.3 Sports commentary / video captioning generation and grounding

- **Tanaka-Ishii, Noda, Frank, Nakashima, Hasida & Matsubara (1998), "MIKE: An Automatic Commentary System for Soccer," ICMAS-98, 285–292.** DOI [10.1109/ICMAS.1998.699067](https://doi.org/10.1109/ICMAS.1998.699067). RoboCup system with six analyzers, importance-scored proposition selection, template natural-language generation, and TTS. Relevance: canonical template/rule-based architecture; separates fact selection from surface realization.
- **Rao, Wu, Liu, Wang & Xie (2024), "MatchTime: Towards Automatic Soccer Game Commentary Generation," EMNLP, 1671–1685.** [ACL](https://aclanthology.org/2024.emnlp-main.99/) · DOI [10.18653/v1/2024.emnlp-main.99](https://doi.org/10.18653/v1/2024.emnlp-main.99). Introduces SN-Caption-test-align (49 manually timestamped matches), a temporal alignment pipeline, the MatchTime dataset, and MatchVoice. Relevance: strongest open soccer commentary dataset/model; shows alignment quality drives generation quality.
- **Mkhallati, Cioppa, Giancola, Ghanem & Van Droogenbroeck (2023), "SoccerNet-Caption: Dense Video Captioning for Soccer Broadcasts Commentaries," CVPR Workshops, 5074–5085.** [CVF](https://openaccess.thecvf.com/content/CVPR2023W/CVSports/html/Mkhallati_SoccerNet-Caption_Dense_Video_Captioning_for_Soccer_Broadcasts_Commentaries_CVPRW_2023_paper.html) · [arXiv:2304.04565](https://arxiv.org/abs/2304.04565). The paper reports almost 37k timestamped commentaries across 715.9 hours; single-anchor dense captioning with METEOR in a ±15 s window. Relevance: event-anchored commentary corpus and benchmark.
- **LLM/RAG commentary:** **You, Huang, Xie, et al. (2025), "TimeSoccer: An End-to-End Multimodal Large Language Model for Soccer Commentary Generation," ACM Multimedia, 3418–3427.** [arXiv](https://arxiv.org/abs/2504.17365) · DOI [10.1145/3746027.3755077](https://doi.org/10.1145/3746027.3755077). Full 45-minute match input with MoFA-Select frame compression and single-anchor dense captioning. **Li, He, Zu, et al. (2025), "Multi-Modal Large Language Model with RAG Strategies in Soccer Commentary Generation," WACV, 6197–6206.** [CVF](https://openaccess.thecvf.com/content/WACV2025/html/Li_Multi-Modal_Large_Language_Model_with_RAG_Strategies_in_Soccer_Commentary_WACV_2025_paper.html) · DOI [10.1109/WACV61041.2025.00604](https://doi.org/10.1109/WACV61041.2025.00604). SoccerComment combines LanguageBind encoders, Vicuna-7B, and retrieval-augmented in-context learning. Relevance: long-form and retrieval-grounded LLM commentary, but neither uses a deterministic fact validator.
- **Hallucination/faithfulness:** **Ji, Lee, Frieske, et al. (2023), "Survey of Hallucination in Natural Language Generation," ACM Computing Surveys 55(12), 1–38.** DOI [10.1145/3571730](https://doi.org/10.1145/3571730). **Min, Krishna, Lyu, et al. (2023), "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation," EMNLP, 12076–12100.** [ACL](https://aclanthology.org/2023.emnlp-main.741/) · DOI [10.18653/v1/2023.emnlp-main.741](https://doi.org/10.18653/v1/2023.emnlp-main.741). **Lewis, Perez, Piktus, et al. (2020), "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," NeurIPS.** [arXiv](https://arxiv.org/abs/2005.11401). Relevance: defines hallucination types, atomic fact precision, and retrieval grounding; motivates the validator design.
- **Ravuru (2025), "Commentary Generation for Soccer Highlights," arXiv:2508.07543.** [arXiv](https://arxiv.org/abs/2508.07543). Extends MatchVoice to GOAL highlights, reproduces MatchTime, and studies training configurations, window size, and hardware limitations. Relevance: direct evidence that commentary training is compute-constrained; supports the small-GPU design. Preprint.

### 2.4 Speech synthesis and prosody for sports commentary

- **Shen, Pang, Weiss, et al. (2018), "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions," ICASSP.** [arXiv](https://arxiv.org/abs/1712.05884) · DOI [10.1109/ICASSP.2018.8461368](https://doi.org/10.1109/ICASSP.2018.8461368). Tacotron 2 reached MOS 4.53 versus 4.58 for professional speech, but had no style control and was not real-time. Relevance: neural TTS naturalness baseline.
- **Ren, Hu, Tan, Qin, Zhao, Zhao & Liu (2021), "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech," ICLR.** [arXiv](https://arxiv.org/abs/2006.04558). Non-autoregressive model with duration, pitch, and energy prediction; 3x faster training than FastSpeech. Relevance: explicit prosody control and low-latency synthesis.
- **Kim, Kong & Son (2021), "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech," ICML, 5530–5540.** [arXiv](https://arxiv.org/abs/2106.06103). VITS combines a conditional VAE, normalizing flows, GAN training, and stochastic duration prediction for parallel waveform generation. Relevance: high-quality open TTS with a small inference footprint.
- **Li, Han, Raghavan, Mischler & Mesgarani (2023), "StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models," NeurIPS.** [arXiv](https://arxiv.org/abs/2306.07691). Style diffusion enables style selection without reference audio; reported human-level naturalness on LJSpeech and VCTK. Relevance: style/excitement control for event-conditioned commentary.
- **Xie, Yan, Zhang & Liu (2024), "Towards Controllable Speech Synthesis in the Era of Large Language Models: A Survey," arXiv:2412.06602.** [arXiv](https://arxiv.org/abs/2412.06602). Taxonomy of emotion, prosody, timbre, and duration control. Relevance: maps controllable TTS dimensions onto commentary excitement.
- **Fish Audio (2025), "Launching Fish Audio S1: A Frontier Text-to-Speech Audio Foundation Model."** [Official blog](https://fish.audio/blog/introducing-s1/). S1 (4B) and S1-mini (0.5B), 2M+ hours of audio, 50+ emotion/tone markers, 13 languages, about 1:7 RTF on an RTX 4090. Relevance: production TTS with emotion tags and interactive latency. Vendor report, not peer-reviewed; no sports-specific prosody evaluation.

### 2.5 Datasets and benchmarks

- **SoccerNet (2018) / SoccerNet-v2 (2021).** 500 broadcast matches, 764 hours, three event classes, then about 300k annotations and 17 classes; action spotting, camera-shot segmentation, replay grounding. [soccer-net.org](https://www.soccer-net.org/). Relevance: core broadcast football benchmark.
- **SoccerNet-Caption (2023).** Almost 37k timestamped commentaries across 715.9 hours; anonymized player/team names; single-anchor dense captioning. Relevance: commentary-specific supervision.
- **MatchTime (2024).** Aligned SoccerNet commentary with SN-Caption-test-align (49 matches) and MatchVoice training data. Relevance: higher-quality video-text alignment than raw SoccerNet-Caption.
- **SoccerReplay-1988 (2025).** 1,988 complete matches, 1,488/250/250 train/valid/test split, six European competitions, 24 event classes; introduced with MatchVision in "Towards Universal Soccer Video Understanding," CVPR 2025, 8384–8394. [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Rao_Towards_Universal_Soccer_Video_Understanding_CVPR_2025_paper.html) · [arXiv:2412.01820](https://arxiv.org/abs/2412.01820) · [Hugging Face (gated)](https://huggingface.co/datasets/Homie0609/SoccerReplay-1988). Relevance: largest multi-modal soccer corpus; direct training source for commentary models.
- **GOAL.** Suglia, Lopes, Bastianelli, et al. (2022), "Going for GOAL: A Resource for Grounded Football Commentaries," arXiv:2211.04534, provides highlight videos with live commentary and baselines for retrieval and play-by-play generation. Qi, Yu, Tu, et al. (2023), "GOAL: A Challenging Knowledge-grounded Video Captioning Benchmark for Real-time Soccer Commentary Generation," CIKM, 5391–5395, DOI [10.1145/3583780.3615120](https://doi.org/10.1145/3583780.3615120), adds about 8,900 clips, 22k sentences, and 42k knowledge triples. Relevance: grounded commentary with external knowledge.
- **SportsMOT (2023), SoccerNet-Tracking (2022), Roboflow football-players.** SportsMOT supplies 240 multi-sport sequences; SoccerNet-Tracking supplies soccer boxes/tracklet IDs. The Roboflow "football-players-detection" dataset provides 663 images under CC BY 4.0; a widely used four-class variant covers Ball, Referee, Football Players, and Goalkeepers. [Roboflow Universe](https://universe.roboflow.com/roboflow-uc1d1/football-players-detection). Relevance: detection/tracking training and evaluation data.

### 2.6 Evaluation metrics

- **Action spotting:** SoccerNet Average-mAP at loose (5–60 s) and tight (1–5 s) tolerances; tight Average-mAP is the discriminative metric for commentary timing (SoccerNet 2018; SoccerNet 2023 Challenges Results).
- **Tracking:** HOTA (geometric mean of detection accuracy and association accuracy; Luiten et al., IJCV 129(2), 548–578, 2020, DOI [10.1007/s11263-020-01375-2](https://doi.org/10.1007/s11263-020-01375-2)); MOTA (CLEAR MOT; Bernardin & Stiefelhagen, EURASIP JIVP, 2008, DOI [10.1155/2008/246309](https://doi.org/10.1155/2008/246309)); IDF1 (Ristani et al., ECCV Workshops, 17–35, 2016, DOI [10.1007/978-3-319-48881-3_2](https://doi.org/10.1007/978-3-319-48881-3_2)).
- **Captioning:** BLEU (Papineni et al., ACL 2002, 311–318, [ACL](https://aclanthology.org/P02-1040/)); ROUGE (Lin, 2004, 74–81, [ACL](https://aclanthology.org/W04-1013/)); METEOR (Banerjee & Lavie, 2005, 65–72, [ACL](https://aclanthology.org/W05-0909/)); CIDEr (Vedantam, Zitnick & Parikh, CVPR 2015, 4566–4575, [CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Vedantam_CIDEr_Consensus-Based_Image_2015_CVPR_paper.html)). Relevance: overlap/fluency metrics; none measures factual grounding.
- **Grounding/faithfulness:** FActScore-style atomic-fact precision against a knowledge source; hallucination taxonomy from Ji et al.; retrieval grounding from RAG. Relevance: the validator can score each generated claim against the deterministic fact layer.
- **TTS quality:** MOS via ITU-T P.800 (08/96) five-point absolute category rating, [ITU](https://www.itu.int/rec/T-REC-P.800/en); supplemented by CMOS, speaker similarity, WER/CER, and RTF.
- **Efficiency:** RTF = synthesis time / audio duration; RTF < 1 is real-time capable. Fish Audio reports about 1:7 RTF on an RTX 4090; FastSpeech 2 and VITS target parallel/low-latency inference. Relevance: determines whether commentary can overlay live match audio.
- **MLOps:** ClearML Task documentation, [clear.ml](https://clear.ml/docs/latest/docs/fundamentals/task/), and auto-logging guide, [clear.ml](https://clear.ml/docs/latest/docs/getting_started/auto_log_exp/). Tasks auto-capture Git state, environment, hyperparameters, scalars, models, and artifacts. Relevance: reproducibility layer for the five-stage pipeline. Official documentation, not a research paper.

---

## 3. Comparison table

| Work (year) | Task | Data | Method | Metric | Limitation |
|---|---|---|---|---|---|
| CALF (2020) | Action spotting | SoccerNet | Context-aware loss around anchors | Average-mAP | Pre-extracted features; no language output |
| E2E-Spot (2022) | Precise event spotting | SoccerNet-v2, fine-grained sports | End-to-end 100-frame clip model | Tight Average-mAP | Short clips; no player/team facts |
| ByteTrack (2022) | Multi-object tracking | MOT17/20 | Low-score box association | MOTA/IDF1/HOTA | Generic MOT; no team ID or possession |
| SportsMOT (2023) | Sports MOT benchmark | 240 sequences, 3 sports | MixSort baseline | HOTA/MOTA/IDF1 | Overhead sports footage, not broadcast commentary |
| SoccerNet-Tracking (2022) | Soccer MOT | 200×30 s + 45-min half | Tracklet annotations and baselines | MOT metrics | Short sequences; no semantic facts |
| Guo et al. (2026) | Detection, team ID, possession | 20 full-match clips | YOLOv8 + DBSCAN jersey clustering + Norfair | RMSE 4.87 | Small evaluation; no commentary/TTS |
| MatchTime (2024) | Commentary generation | SoccerNet + 49 aligned matches | MatchVoice: visual encoder + Perceiver + LLM | Captioning quality + alignment benchmark | No explicit fact validator; residual alignment noise |
| TimeSoccer (2025) | Full-match commentary | SoccerNet full matches | TimeChat + MoFA-Select frame compression | SDVC captioning + temporal alignment | Heavy MLLM; not small-GPU |
| SoccerComment (2025) | Commentary + retrieval | SoccerNet | LanguageBind + Vicuna-7B + RAG memory | Captioning + retrieval/zero-shot | RAG only; no deterministic validator |
| Fish Audio S1 (2025) | TTS | 2M+ hours proprietary audio | Qwen3 backbone + codec + RLHF | WER/CER, speaker distance, RTF | Vendor claims; 4B proprietary; no sports prosody eval |

---

## 4. Gaps this project addresses

1. **Grounded long-form commentary from broadcast video.** MatchTime and TimeSoccer generate fluent commentary, but they do not expose a deterministic, auditable fact layer; SoccerNet-Caption is event-anchored short text. The project separates facts (score, minute, zone, possession, tracked players) from surface realization, making long-form commentary inspectable.
2. **Event-to-speech.** TTS research is generic; no peer-reviewed sports-commentary TTS system with event-conditioned prosody was found. Fish Audio S1's emotion/tone tags are the closest production mechanism, but their use for football events is unvalidated. The project maps event type and intensity to TTS style and timing.
3. **Hallucination control.** No dedicated football-commentary faithfulness benchmark exists; FActScore-style atomic validation has not been applied to this domain, and SoccerComment uses retrieval but no validator. The project's grounding validator checks generated claims against the fact layer, enabling precision/recall-style faithfulness reporting.
4. **Small-GPU reproducibility.** TimeSoccer and MatchVision are heavy; Ravuru (2025) explicitly documents hardware limits and window-size effects. GPT-2 with LoRA is a pragmatic alternative: LoRA freezes pretrained weights and injects low-rank matrices, cutting trainable parameters and memory while matching fine-tuning quality on GPT-2 (Hu et al., ICLR 2022, [OpenReview](https://openreview.net/forum?id=nZeVKeeFYf9); Radford et al., 2019, [GPT-2 report](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)). YOLO + ByteTrack keeps the perception stack deployable.
5. **Evaluation and MLOps fragmentation.** Spotting, tracking, captioning, grounding, and TTS papers use disjoint metrics and rarely report experiment-tracking practice. ClearML provides task, artifact, and metric logging for reproducible runs; the project can report a combined evaluation spanning event timing, tracking quality, fact faithfulness, captioning overlap, and TTS MOS/RTF.

---

## 5. Search strategy

**Databases and indexes:** Google Scholar, arXiv, ACL Anthology, CVF Open Access, IEEE Xplore, ACM Digital Library, SpringerLink, DBLP, Semantic Scholar, Hugging Face Datasets, and official project documentation (Ultralytics, ClearML, Fish Audio, ITU, SoccerNet).

**Exact queries used (representative):**
- "SoccerNet scalable dataset action spotting soccer videos CVPR 2018"
- "SoccerNet-v2 dataset benchmarks holistic understanding broadcast soccer videos"
- "CALF context-aware loss action spotting soccer"
- "NetVLAD++ temporally precise action spotting soccer"
- "E2E-Spot temporally precise fine-grained events ECCV 2022"
- "SoccerNet 2023/2024 challenges results action spotting tracking"
- "ByteTrack associating every detection box ECCV 2022"
- "SportsMOT multi-object tracking dataset ICCV 2023"
- "SoccerNet-Tracking multiple object tracking benchmark"
- "jersey color team identification football possession YOLOv8 DBSCAN"
- "MatchTime automatic soccer commentary EMNLP 2024"
- "SoccerNet-Caption dense video captioning"
- "SoccerReplay-1988 dataset Hugging Face"
- "LLM soccer commentary RAG grounding hallucination"
- "TimeSoccer end-to-end multimodal LLM commentary"
- "StyleTTS 2 / VITS / FastSpeech 2 / Tacotron 2"
- "Fish Audio OpenAudio S1 technical report"
- "HOTA / MOTA / IDF1 / CIDEr / FActScore / MOS / RTF"
- "ClearML experiment tracking documentation"

**Inclusion criteria:** peer-reviewed papers, arXiv/ACL/CVF/IEEE/ACM/Springer records, or official documentation; football/soccer broadcast focus, or a general method/benchmark directly applicable to it; English; available URL/DOI; 2015–2026 with essential classics from 1998–2016.

**Exclusion criteria:** non-football sports unless the method/benchmark generalizes (SportsMOT retained for its football subset); opinion pieces, news, and inaccessible pages; unverifiable citations; vendor claims not clearly labelled; duplicate preprint/published versions (published version preferred).

**Date range:** 1998–2026, with emphasis on 2020–2026. Classics retained: MIKE (1998), BLEU (2002), ROUGE (2004), METEOR (2005), CLEAR MOT (2008), YOLO (2016), IDF1 (2016), Tacotron 2 (2017).

**Verification note:** Every citation above was cross-checked against at least one publisher, arXiv, ACL Anthology, CVF, IEEE, ACM, DBLP, or official documentation record. No unverified paper is included. Non-peer-reviewed sources are explicitly labelled: Fish Audio S1 (vendor blog), ClearML (official docs), Ultralytics (software docs), SoccerNet 2023/2024 challenge reports (arXiv), Ravuru 2025 (preprint), GPT-2 (OpenAI technical report), and Roboflow (dataset page).

---

## 6. Reading order and next steps

1. **Start with the task definitions:** SoccerNet (2018) → SoccerNet-v2 (2021) → SoccerNet 2023/2024 results. This fixes the event taxonomy and the tight/loose Average-mAP protocol.
2. **Then the components:** ByteTrack + SoccerNet-Tracking + SportsMOT for tracking; Naik et al. and Guo et al. for jersey/possession; MatchTime + SoccerNet-Caption for commentary data and alignment.
3. **Then the grounding argument:** MIKE (fact selection vs surface realisation) → FActScore + hallucination survey → our validator design.
4. **Then TTS:** Tacotron 2 → FastSpeech 2 / VITS → StyleTTS 2 / controllable-TTS survey → Fish Audio S1 (label as vendor).
5. **To write the paper:** use the comparison table as related-work structure; report our pipeline against the gaps in §4; add an evaluation table with spotting mAP, tracking HOTA/IDF1, captioning BLEU/ROUGE/METEOR/CIDEr, grounding acceptance, and TTS MOS/RTF.

**Known gaps in this review to fill next:**
- A dedicated **sports-commentary TTS** paper/benchmark (none found).
- A **football-commentary faithfulness benchmark** (none found; FActScore is general).
- Independent (non-vendor) **Fish Audio S1** evaluation.
- Full-text verification of the 2026 Sensors possession paper and the 2025 TimeSoccer/SoccerComment camera-ready versions.
