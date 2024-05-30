# ChatGPT Research

## Setup

Use [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
or [mamba](https://mamba.readthedocs.io/en/latest/installation.html).
(Environments were tested with **mamba**)

CPU Environment:
```shell
mamba env create --file env-cpu.yml
mamba activate gpt-gpu-39
```

GPU Environment:
```shell
mamba env create --file env-gpu.yml
mamba activate gpt-gpu-39
```

Or the same commands using `conda`.

## config.py

### Used variables
#### Misc
 * `model_name` : `roberta-base`|`roberta-large` - defines model
 * `faiss_use_gpu`: bool - use GPU when building FAISS index or not
 * `show_plot`: bool - show plot after estimation script run
 * `threshold`: float - Lowest acceptable cosine distance, when comparing embeddings

#### Files
 * `artifacts_folder`: string - folder that contains all the artifacts -
                         files that can be loaded by scripts, and files produced by the scripts
 * `embeddings_file`: string - where embeddings should be saved _\[Not Used\]_
 * `mapping_file`: string - where mapping of embedding index to source is saved (CSV)
 * `index_file`: string - where FAISS index is saved (faiss format)
 * `centroid_file`: sting - where embeddings centroid is saved (Numpy's NPY file)
 * `temp_index_file`: string - where temporary index built from query text is written
 * `source_index_path`: string - where Lucene source index is located 

#### Wiki Articles
 * `page_names`: list\[string\] - list of Wikipedia articles on _target topic_
 * `unrelated_page_names`: list\[string\] - list of Wikipedia articles not on _target topic_,
   * used when estimating thresholds and centroid
 * `unrelated_page_names_2`: list\[string\] - large list of Wikipedia articles not on _target topic_,
   * used for centroid estimation

### Loading from environment

You can set variable as `None` and in this case Config will try to load it from environment.
For example `source_index_path` is set to None and value will be acquired from `SOURCE_INDEX_PATH` environment variable.

### Paths and files

Variable that has `file` or `path` at the end of the name:
 * Must contain path
 * If it's not an absolute path, it must be a path relative to **artifacts folder**

## Run scripts

All runnable scripts placed in `scripts` folder.

When environment is activated you can run python scripts in this project.
To run scripts use:
```shell
# in project root
python -m scripts <script-name> [scipt-args...]
```

### Scripts

and their vague description...

 * `build_index_from_wiki` - builds embedding index from wiki articles
 * `collect_pop_quiz` - surveys ChatGPT for answers on quiz
   * Quiz name should be supplied as first parameter `python -m scripts collect_pop_quiz test_quiz`
   * See details in top comment in the file
   * Requires environment variable `OPENAI_API_KEY` to be set
 * `estimate_centroid` - collects large amount of embeddings from wiki articles and computes centroid estimation
 * `estimate_thresholds` - collects data on related and unrelated topics and estimates threshold
 * `filter_answers` - filters correct answers provided by ChatGPT
   * See details in top comment in the file
 * `auc` - script for get roc-auc graph
 * `algorithm_1` - algorithm without chains
 * `algorithm_2` - algorith with chains
 * `algorithm_1_server` - algorithm without chains for server
 * `algorithm_2_server` - algorithm with chains for server
 * `scrape_wiki_dataset` - scrape data from wiki pages 
 * `survey`
   * See details in top comment in the file
