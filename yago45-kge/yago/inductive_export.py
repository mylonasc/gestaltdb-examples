"""Export a lightweight, ID-table-free inductive GNN for actual deployment."""
from pathlib import Path

import tensorflow as tf

from .inductive_model import complex_score


def export_encoder(model, output_dir):
    """Persist only transferable encoder/decoder resources, never ID tables."""
    excluded = {id(variable) for layer in (model.entity_table, model.base_relations) for variable in layer.variables}
    module = tf.Module(name="inductive_graph_autoencoder")
    module.encoder_variables = [variable.value for variable in model.variables if id(variable) not in excluded]
    feature_dim = int(model.input_projection.kernel.shape[0])
    graph_signature = dict(features=tf.TensorSpec((None, feature_dim), tf.float32),
                           senders=tf.TensorSpec((None,), tf.int32),
                           receivers=tf.TensorSpec((None,), tf.int32),
                           relations=tf.TensorSpec((None,), tf.int32))
    module.encode = tf.function(lambda graph: model.encode(graph), input_signature=[graph_signature])
    module.predict_graph = tf.function(
        lambda graph, triples: model.contextual_scores(model.encode(graph), triples),
        input_signature=[graph_signature, tf.TensorSpec((None, 3), tf.int32)])
    dim = int(model.context_relations.embeddings.shape[1])
    module.score_embeddings = tf.function(
        lambda subjects, relations, objects: complex_score(subjects, model.context_relations(relations), objects),
        input_signature=[tf.TensorSpec((None, dim), tf.float32), tf.TensorSpec((None,), tf.int32),
                         tf.TensorSpec((None, dim), tf.float32)])
    tf.saved_model.save(module, str(Path(output_dir)), signatures={
        "encode": module.encode.get_concrete_function(),
        "predict_graph": module.predict_graph.get_concrete_function(),
        "score_embeddings": module.score_embeddings.get_concrete_function()})
