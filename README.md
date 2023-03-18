# structural-probes-for-russian

Code is heavily based on John Hewitt's [structural-probes](https://github.com/john-hewitt/structural-probes). Reimplemented some stuff either to understand what I'm doing or because it seemed an easier way.

## Demo

Steps:

1. Download [UD Russian SynTagRus](https://universaldependencies.org/treebanks/ru_syntagrus/index.html#ud-russian-syntagrus) (train, dev and test splits) to your current directory.
2. [Install](https://bert-as-service.readthedocs.io/en/latest/section/get-start.html#installation) `bert-as-service`.
3. Start server. Example command, where PATH_TO_BERT is path to the directory with a BERT-like model:

`bert-serving-start -pooling_strategy NONE -max_seq_len NONE -show_tokens_to_client -cpu -pooling_layer -6 -model_dir PATH_TO_BERT`

[Adjust parameters](https://bert-as-service.readthedocs.io/en/latest/source/server.html#BERT%20Parameters) as you wish, but keep `-pooling_strategy NONE`. 7 (aka -6) is presumably the best performing layer (syntax-wise).

4. Run `scripts/conllu_to_bert.py`. Example command:

`py scripts/conllu_to_bert.py --conllu_dir "path/to/dir/with/conllu/files" --bert_dir "path/to/dir/with/bert/model" --bert_alias "rubert"`

An hdf5 file with embedded conllu dataset will be created at `.embeddings/rubert`.

5. Create config file at `config/one_of_two_task_aliases/preferably_telling_config_name.yaml` where `one_of_two_task_aliases` is either `prd` (parse-distance probing task) or `pad` (parse-depth probing task). For some reason I used those closely following original notation.

6. Run `demo.py` to train a probe, make predictions and report metrics on the test set. Example command:

`py demo.py --config_path config/prd/str-prd-rubert-1.yaml`,

where `str-prd-rubert-1.yaml` is the demo config or your config.
