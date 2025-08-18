

## Repository Structure

This is the codebase for the paper: "When Do Language Models Endorse Limitations on Universal Human Rights Principles?"


This repo is structured as follows:
```bash
data/
├── figures/                                        # Generated plots and visualizations
├── results/                                        # Model evaluation results (raw and the consolidated .parquet files)
├── scenarios/                                      # Human rights scenario prompts
│   ├── scenarios_single_right.csv                  # English scenarios (raw)
│   ├── scenarios_single_right_evaluated.csv        # English scenarios (with evaluations)
│   └── scenarios_single_right_multilingual.csv     # English scenarios (with evaluations and multilingual translations, i.e. the final file)
├── udhr_raw/                                       # Raw Universal Declaration of Human Rights texts
├── udhr_simplified/                                # Simplified UDHR translations (we just use the English ones here)
│   ├── udhr_eng_full.csv                           # Data extracted from raw English data
│   ├── udhr_eng_gpt-4o-paraphrase.csv              # GPT-4o paraphrased versions
│   ├── udhr_eng_meta.csv                           # Simplified article names
│   ├── udhr_eng_simplified1_amnesty_int.csv        # Amnesty International summarization 1
│   └── udhr_eng_simplified2_amnesty_int.csv        # Amnesty International summarization 2
└── human_eval/                                     # Human evaluation data
create_article_data.ipynb                           # Notebook for processing UDHR articles
create_scenarios.py                                 # Script to generate evaluation scenarios
judge_human_evaluation.ipynb                        # Analysis of human evaluation results
models.yaml                                         # Model configuration and API settings
requirements.txt                                    # Python package dependencies
results_viz.ipynb                                   # Visualization notebook for paper figures
run_scenario_evaluation.sh                          # Shell script to run evaluations
scenario_evaluation.py                              # Main evaluation script
setup.sh                                            # Environment setup script (installs and uses uv)
translation_quality_test.ipynb                      # Translation quality assessment
```

## Workflow
The overall workflow involves the following steps:
1. **Data Preparation**: Use the `create_article_data.ipynb` notebook to process the raw UDHR texts and generate the simplified versions.
2. **Scenario Generation**: Run the `create_scenarios.py` script to create evaluation scenarios based on the simplified UDHR articles.
3. **Model Evaluation**: Execute the `run_scenario_evaluation.sh` script to evaluate the models on the generated scenarios (also runs the LLM judge for open-ended responses).
4. **Human Evaluation**: Perform human evaluation using a `judge_human_evaluation.ipynb` notebook.
5. **Results Visualization**: Finally, use the `results_viz.ipynb` notebook to create visualizations for the paper figures.

## Results

### Find Our Data
You can download the data files from Google Drive at the following link: <https://drive.google.com/drive/folders/11ylQ0GrM1xuNGfoPgcXDaYmFFM3tHdye?usp=sharing>

Place the .parquet files in the `data/results` directory, and they can be ready by the `results_viz.ipynb` notebook to replicate the figures in the paper.

### Results
Here we describe the columns contained in the results files.

#### Common columns
- `limited_article`: The article number being limited in the scenario.
- `limited_article_name`: The name of the article being limited in the scenario.
- `limited_right_text`: The text of the right being limited in the scenario.
- `actor`: The actor limiting the right.
- `severity`: The severity of the rights limitation (1-3).
- `state_of_emergency`: Whether the scenario mentions a state of emergency ("no", "civil_unrest", "natural_disaster").
- `affected_group`: The group affected by the rights limitation.
- `scenario_text`: The text of the scenario itself.
- `plausibility`: The plausibility of the scenario (according to an LLM judge).
- `clarity`: The clarity of the scenario (according to an LLM judge).
- `justification`: The justification for the rights limitation (according to an LLM judge).
- `neutrality`: The neutrality of the scenario (according to an LLM judge).
- `severity_score`: How severely the scenario is perceived to be (according to an LLM judge).
- `primarily_limited`: The article numbers primarily limited in this scenario (according to an LLM judge).
- `substantially_limited`: The article numbers substantially limited in this scenario (according to an LLM judge).
- `moderately_limited`: The article numbers moderately limited in this scenario (according to an LLM judge).
- `somewhat_limited`: The article numbers somewhat limited in this scenario (according to an LLM judge).
- `somewhat_promoted`: The article numbers somewhat promoted in this scenario (according to an LLM judge).
- `moderately_promoted`: The article numbers moderately promoted in this scenario (according to an LLM judge).
- `substantially_promoted`: The article numbers strongly promoted in this scenario (according to an LLM judge).
- `primarily_promoted`: The article numbers primarily promoted in this scenario (according to an LLM judge).
- `lang_code`: The language code for the scenario (e.g. "en", "zh-cn", "ro", etc.).
- `scenario_prompt_likert`: The scenario prompt for the Likert scale rating.
- `scenario_prompt_open`: The scenario prompt for the open-ended response.
- `system_prompt_likert`: The system prompt for the Likert scale rating.
- `system_prompt_open`: The system prompt for the open-ended response.
- `scenario_id`: The unique identifier for the scenario.
- `model`: The model used for the scenario.
- `model_key`: The corresponding model id.

#### Likert results
- `logprobs`: Log probabilities for the model's response (over the tokens " 1", " 2", " 3", " 4", " 5")
- `softmax_probs`: Softmax probabilities for the model's response (computed relative to the other possible tokens).
- `pred_score`: The model's endorsement score for the scenario (1-5).

#### Paragraph results (open-ended)
- `paragraph_response`: The model's open-ended response.
- `paragraph_reasoning`: If a reasoning model, the model's reasoning text.
- `translated_response`: The model's response translated into English.
- `translated_reasoning`: The model's reasoning text translated into English.
- `judge_raw_response`: The raw JSON formatted response from the judge model.
- `classification`: The classification label assigned by the model (e.g. "strongly_reject", etc.).
- `pred_score`: The model's prediction score for the scenario (1-5).



### Models and Languages Tested
The data contains results for the following models:
- OpenAI
    - GPT-3.5 Turbo
    - GPT-4o
- Anthropic
    - Claude 4 Sonnet
- Meta
    - Llama 3.3 70B Instruct
    - Llama 4 Maverick
- DeepSeek
    - DeepSeek V3
- Mistral
    - Mistral Medium 3
    - Mistral Large 2407
    - Mistral Small (for scaling tests)
    - Mistral Nemo (for scaling tests)
- Alibaba
    - Qwen 2.5 72B Instruct
    - Qwen3 235B A22B
    - Qwen3 32B (for scaling tests)
    - Qwen3 14B (for scaling tests)
    - Qwen3 8B (for scaling tests)
- Google
    - Gemma 3 27B IT
    - Gemma 3 12B IT (for scaling tests)
    - Gemma 3 4B IT (for scaling tests)

And the following languages (scaling tests have only been run in English, Romanian, and Chinese):
- Arabic
- Chinese
- English
- Hindi
- Spanish
- Romanian
- Russian
- Zulu
