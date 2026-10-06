# Sources

## Problem evidence
- Workplace harassment and violence in the public sector: integrative literature review — https://pmc.ncbi.nlm.nih.gov/articles/PMC13034581/
- PSHRA, civility declining in public interactions with government employees — https://pshra.org/research-finds-civility-declining-in-public-interactions-with-government-employees/
- Customer aggression and emotion regulation of service employees — http://homepages.se.edu/cvonbergen/files/2013/01/The-Customer-is-Not-Always-Right-Customer-Aggression-and-Emotion-Regulation-of-Service-Employees.pdf
- 911 dispatcher mental-health data — https://www.centralsquare.com/resources/articles/911-dispatcher-mental-health-data

## Data
- NYC 311 Service Requests (2020–present) — https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9
- Jigsaw / Civil Comments (google/civil_comments) — https://arxiv.org/abs/2301.11125
- CREMA-D (Cao et al., 2014), via confit/cremad-parquet on Hugging Face (ODbL)

## Methods this prototype builds on
- Waseem et al. (2017), *Understanding Abuse: A Typology of Abusive Language Detection Subtasks* — directed vs. generalised abuse → directedness gate (v3)
- Dale et al. (2021), *Text Detoxification using Large Pre-trained Neural Models* (condBERT) — token-level masking → extractive detox renderer (v2)
- Laugier et al. (2021), *Civil Rephrases of Toxic Texts with Self-Supervised Transformers* — prior work on rephrasing
- Logacheva et al. (2022), *ParaDetox* — parallel detox corpus / BART detox model
- Hanu & Unitary (2020), *Detoxify* — toxic-bert / unbiased-toxic-roberta weights
- Yin, Hay & Roth (2019), *Benchmarking Zero-shot Text Classification* — NLI-as-classifier safety net (v6)
- Troiano et al. (2018), *A Computational Exploration of Exaggeration* — hyperbole/figurative venting filter (v8)
- Tunstall et al. (2022), *SetFit* — few-shot nearest-prototype intent classification
- Eyben et al. (2016), *eGeMAPS* — arousal-relevant prosodic features
- Macháček, Dabre & Bojar (2023), *Turning Whisper into Real-Time Transcription System* — LocalAgreement streaming
- Sainath et al. (2019), *Two-Pass End-to-End Speech Recognition* — streaming first pass + second-pass rescoring idea
