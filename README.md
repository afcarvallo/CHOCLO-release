# CHOCLO: supplementary data and code (anonymous submission)

CHOCLO is an entity-centered benchmark of culturally grounded factual knowledge about
Latin America, with Europe and the USA as contrastive regions. All questions and answers
are in Spanish.

| Region | Entities | Q/A pairs |
|---|---:|---:|
| LATAM (18 countries) | 34,818 | 104,847 |
| Europe | 17,401 | 52,195 |
| USA | 15,385 | 46,160 |

## Contents

```
data/
  benchmark/           choclo_{latam,europe,usa}.csv  question, reference answer, entity, country, category, difficulty
  model_outputs/       Score_<model>_<region>.csv.gz  answers and scores of DeepSeek-V3.1, GPT-4o-mini,
                                                      Gemma-3-4B, Qwen2.5-7B (lexical, embedding, LLM judge)
  gpt55_subsample/     GPT-5.5 and the four models on a stratified 9,450-question subsample, re-judged
  human_validation/    1,832 judgments of 1,021 LATAM Q/A pairs by 17 annotators (A01-A17)
  human_judge/         120 model answers rated by 3 annotators (H1-H3) with the LLM-judge score
  popularity/          Spanish Wikipedia page views (all regions) and web hit counts (LATAM)
  qa_validation_regions/  150 Q/A pairs balanced across regions, labeled by 3 annotators (V1-V3)
  relevance_audit/     210 entities (70 per region), cultural-relevance labels by 3 annotators (R1-R3)
  triplets_sample/     subject-relation-object triplets for the Q/A pairs of a stratified sample of
                       2,100 entities (100 per region x category), used for the KG and multi-hop analyses
  additional_judges/   scores of two further judges (gpt-4o-2024-08-06 and the open-weight
                       Llama-3.3-70B-Instruct) on a stratified subsample and on the 120 human-rated answers
code/                  analysis scripts (Python 3.10) and the annotation app
  gpu_probe/           hidden-state probes of Gemma-3-4B and Qwen2.5-7B (needs a GPU), with its input bundle
results/               CSV outputs behind every table and figure in the paper (E0-E24)
```

Annotators are identified only by anonymous IDs. Entity names come from Wikipedia and Wikidata.

## Reproducing the analyses

```bash
pip install -r code/requirements.txt
mkdir -p code/data && cp data/model_outputs/*.csv.gz code/data/
cp data/popularity/web_hits_latam.csv code/data/   # used by the popularity analysis
cd code
python experiments.py      # E0-E11: regional, composition, category, difficulty, popularity, judge analyses
python make_tables.py      # LaTeX tables
```

### Analyses added in the revision

| ID | Script | Analysis |
|---|---|---|
| E18 | `validation_regions.py` | Q/A validation balanced across regions |
| E19 | `relevance_audit.py` | cultural relevance of sampled entities |
| E20 | `triplets_from_qa.py` | triplet structure of the questions (relations, connectivity) |
| E21 | `multihop_models.py` | single-relation, compositional, and two-hop questions by model |
| E22-E23 | `second_judge.py` | additional judges (`--judge meta-llama/llama-3.3-70b-instruct --provider openrouter --tag E23_open_judge`) |
| E24 | `gpu_probe/extract_hidden.py`, `gpu_probe/probe_hidden.py` | linear probes on the evaluated models' own hidden states, including KEEN-style entity tokens |

The hidden-state probe runs on a GPU machine: `python gpu_probe/extract_hidden.py --model google/gemma-3-4b-it --tag gemma`
followed by `python gpu_probe/probe_hidden.py --tag gemma --target Gemma-3-4B` (and the same with
`Qwen/Qwen2.5-7B-Instruct`, tag `qwen`). Its inputs are in `code/gpu_probe/bundle/`.

`openai_eval.py` (GPT-5.5 subsample) and `embeddings.py` (text-embedding-3-large entity
embeddings, used by `probe.py`) call the OpenAI API and need `OPENAI_API_KEY`; their outputs
are already included in `data/` and `results/`. `pageviews.py` queries the Wikimedia API.
`triplets_from_qa.py` and `second_judge.py` also call LLM APIs (OpenAI / OpenRouter); their outputs are
included in `data/triplets_sample/`, `data/additional_judges/`, and `results/`.

## Column reference (model outputs)

`entidad` entity · `pais` country · `categoria` category · `dificultad` difficulty
(FÁCIL / INTERMEDIA / DIFÍCIL) · `pregunta` question · `respuesta` reference answer ·
`respuesta_gpt5` model answer · `score_lexico` overlap score · `score_embedding` embedding
similarity · `score_gpt` LLM-judge score (GPT-5-mini, graded 0-1). The paper's lexical metric
(normalized token F1) is computed by `code/common.py`.

## License

Data: MIT (derived from Wikipedia/Wikidata content under their respective licenses). Code: MIT.
