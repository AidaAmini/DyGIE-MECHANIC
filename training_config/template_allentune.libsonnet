{
  local DROPOUT = std.parseJson(std.extVar("DROPOUT")),
  local LEARNING_RATE = std.parseJson(std.extVar("LEARNING_RATE")),
  local HIDDEN_SIZE = std.parseInt(std.extVar("HIDDEN_SIZE")),
  local CUDA_VISIBLE_DEVICES = std.extVar("CUDA_VISIBLE_DEVICES"),
  local master_port = std.parseInt(std.extVar("master_port")),

  DyGIE: {
    local dygie = self,

    // Mapping from target task to the metric used to assess performance on that task.
    local validation_metrics = {
      'ner': '+MEAN__ner_f1',
      'relation': '+MEAN__relation_f1',
      'coref': '+MEAN__coref_f1',
      'events': '+arg_class_f1'
    },

    ////////////////////

    // REQUIRED VALUES. Must be set by child class.

    // Paths to train, dev, and test data.
    data_paths :: error 'Must override `data_paths`',

    // Weights on the losses of the model components (e.g. NER, relation, etc).
    loss_weights :: error 'Must override `loss_weights`',

    // Make early stopping decisions based on performance for this task.
    // Options are: `['ner', 'relation', 'coref': 'events']`
    target_task :: error 'Must override `target_task`',

    // DEFAULT VALUES. May be set by child class..
    bert_model :: 'bert-base-cased',
    // If using a different BERT, this number may be different. It's up to the user to set the
    // appropriate value.
    max_wordpieces_per_sentence :: 512,
    max_span_width :: 12,
    cuda_device :: -1,

    ////////////////////

    // All remaining values can be overridden using the `:+` mechanism
    // described in `doc/config.md`
    random_seed: 13370,
    numpy_seed: 1337,
    pytorch_seed: 133,
    dataset_reader: {
      type: 'dygie',
      token_indexers: {
        bert: {
          type: 'pretrained_transformer_mismatched',
          model_name: dygie.bert_model,
          max_length: dygie.max_wordpieces_per_sentence
        },
      },
      max_span_width: dygie.max_span_width,
      cache_directory: 'cache',
    },
    train_data_path: std.extVar("ie_train_data_path"),
    validation_data_path: std.extVar("ie_dev_data_path"),
    test_data_path: std.extVar("ie_test_data_path"),
    // If provided, use pre-defined vocabulary. Else compute on the fly.
    model: {
      type: 'dygie',
      embedder: {
        token_embedders: {
          bert: {
            type: 'pretrained_transformer_mismatched',
            model_name: dygie.bert_model,
            max_length: dygie.max_wordpieces_per_sentence
          },
        },
      },
      initializer: {  // Initializer for shared span representations.
        regexes:
          [['_span_width_embedding.weight', { type: 'xavier_normal' }]],
      },
      module_initializer: {  // Initializer for component module weights.
        regexes:
          [
            ['.*weight', { type: 'xavier_normal' }],
            ['.*weight_matrix', { type: 'xavier_normal' }],
          ],
      },
      loss_weights: dygie.loss_weights,
      feature_size: 20,
      max_span_width: dygie.max_span_width,
      target_task: dygie.target_task,
      feedforward_params: {
        num_layers: 2,
        hidden_dims: HIDDEN_SIZE,
        dropout: DROPOUT,
      },
      modules: {
        coref: {
          spans_per_word: 0.3,
          max_antecedents: 100,
          coref_prop: 0,
        },
        ner: {},
        relation: {
          spans_per_word: 0.5,
        },
        events: {
          trigger_spans_per_word: 0.3,
          argument_spans_per_word: 0.8,
          loss_weights: {
            trigger: 0.2,
            arguments: 1.0,
          },
        },
      },
    },
    data_loader: {
      type: 'ie_batch',
      batch_size: 1,
    },


   trainer: {
      "distributed": false,
      "cuda_device": 0,
	

      checkpointer: {
        num_serialized_models_to_keep: 3,
      },
      num_epochs: 50,
      grad_norm: 5.0,
      validation_metric: validation_metrics[dygie.target_task],
      optimizer: {
        type: 'adamw',
        lr: LEARNING_RATE,
        weight_decay: 0.0,
        parameter_groups: [
          [
            ['_embedder'],
            {
              lr: 5e-5,
              weight_decay: 0.01,
              finetune: true,
            },
          ],
        ],
      },
      learning_rate_scheduler: {
        type: 'slanted_triangular'
      }
    },
  },
}
