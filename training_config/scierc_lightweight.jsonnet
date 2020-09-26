local template = import "./training_config/template.libsonnet";

template.DyGIE {
  bert_model: "allenai/biomed_roberta_base",
  cuda_device: 0,
  loss_weights: {
    ner: 1.0,
    relation: 1.0,
    coref: 0.0,
    events: 0.0
  },
  target_task: "relation"
}
