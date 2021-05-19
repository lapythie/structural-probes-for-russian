# structural-probes-for-russian

Code is heavily based on John Hewitt's [structural-probes](https://github.com/john-hewitt/structural-probes). Reimplemented some stuff either to understand what I'm doing or because it seemed an easier way.

## Steps

0. [Install](https://bert-as-service.readthedocs.io/en/latest/section/get-start.html#installation) `bert-as-service`.
1. Start server. Example command, where PATH_TO_BERT is path to the directory with a BERT-like model:

`bert-serving-start -pooling_strategy NONE -max_seq_len NONE -show_tokens_to_client -cpu -pooling_layer -6 -model_dir PATH_TO_BERT`

[Adjust parameters](https://bert-as-service.readthedocs.io/en/latest/source/server.html#BERT%20Parameters) as you wish, but keep `-pooling_strategy NONE`.

7 (aka -6) is presumably the best performing layer (syntax-wise).
