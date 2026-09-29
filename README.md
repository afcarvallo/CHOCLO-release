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
code/                  analysis scripts (Python 3.10) and the annotation app
results/               CSV outputs behind every table and figure in the paper (E0-E16)
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

`openai_eval.py` (GPT-5.5 subsample) and `embeddings.py` (text-embedding-3-large entity
embeddings, used by `probe.py`) call the OpenAI API and need `OPENAI_API_KEY`; their outputs
are already included in `data/` and `results/`. `pageviews.py` queries the Wikimedia API.

## Column reference (model outputs)

`entidad` entity · `pais` country · `categoria` category · `dificultad` difficulty
(FÁCIL / INTERMEDIA / DIFÍCIL) · `pregunta` question · `respuesta` reference answer ·
`respuesta_gpt5` model answer · `score_lexico` overlap score · `score_embedding` embedding
similarity · `score_gpt` LLM-judge score (GPT-5-mini, graded 0-1). The paper's lexical metric
(normalized token F1) is computed by `code/common.py`.

## License

Data: MIT (derived from Wikipedia/Wikidata content under their respective licenses). Code: MIT.
