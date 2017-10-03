from object_net import object_net_components
from object_net import object_net_writer
from object_net import padder
from object_net import types
import numpy as np
import tensorflow as tf
import tf_utils


def train(args, data: np.array, layer_size: int, num_layers: int):
    state_output_pairs_data = map(sequence_type.get_state_output_pairs, data)
    padded_data = padder.PaddedData.from_unpadded(state_output_pairs_data)
    object_net_data_placeholder = padder.PlaceholderPaddedData()
    data_holder = tf_utils.data_holder.DataHolder(args, lambda i: padded_data[i], len(padded_data))
    model_input = tf.zeros([object_net_data_placeholder.batch_size, 1])

    # Set up object_net model
    def get_model(training: bool) -> object_net_writer.ObjectNetWriter:
        return object_net_writer.ObjectNetWriter(
            truth_padded_data=object_net_data_placeholder,
            initial_hidden_vector_input=model_input,
            object_type=sequence_type,
            training=training,
            hidden_vector_network=object_net_components.LstmHiddenVectorNetwork(
                layer_size, num_layers, object_net_components.AdditionHiddenVectorCombiner()))

    model = get_model(True)
    model_test = get_model(False)
    optimizer = tf.train.AdamOptimizer().minimize(model.cost)
    summary = tf.summary.scalar("cost", model.cost)

    # Run training
    def train_step(session, step, training_input, _, summary_writer, __):
        _, _, summaries = session.run(
            [optimizer, model.cost, summary],
            object_net_data_placeholder.get_feed_dict(training_input))

        summary_writer.add_summary(summaries, step)

    def test_step(session, step, testing_input, _, summary_writer, __):
        cost_result, all_summaries = session.run(
            [model.cost, summary],
            object_net_data_placeholder.get_feed_dict(testing_input))

        summary_writer.add_summary(all_summaries, step)

        print("Test cost at step %d: %f" % (step, cost_result))

        show_examples(session, testing_input)

    def show_examples(session, model_input):
        # Limit to 10 inputs
        model_input = [x[:10] for x in model_input]

        generated_states_padded, \
        generated_outputs_padded, \
        generated_outputs_counts_padded, \
        generated_step_counts = session.run(
            [
                model_test.generated_states_padded,
                model_test.generated_outputs_padded,
                model_test.generated_outputs_counts_padded,
                model_test.generated_step_counts],
            object_net_data_placeholder.get_feed_dict(model_input))

        copied_testing_input = padder.PaddedData(
            generated_step_counts, generated_outputs_counts_padded, generated_states_padded, generated_outputs_padded)
        unpadded = padder.unpad(copied_testing_input)

        def try_array_to_value(_array):
            try:
                return list(sequence_type.get_value_from_state_output_pairs(_array))
            except StopIteration:
                return [-1]

        generated_sequences = [try_array_to_value(array) for array in unpadded]

        [print(s) for s in generated_sequences]

    tf_utils.generic_runner.run_with_test_train_steps(
        args,
        "rnn_object_net_comparison",
        get_batch_fn=lambda size: (data_holder.get_batch(size), None),
        testing_data=(data_holder.get_test_data(), None),
        test_step_fn=test_step,
        train_step_fn=train_step)


def __get_rnn_model(layer_size: int, num_layers: int, model_input: tf.Tensor) -> tf.Tensor:
    def get_cell(index: int):
        with tf.variable_scope("rnn_cell_%d" % index):
            return tf.nn.rnn_cell.LSTMCell(layer_size, state_is_tuple=False)

    batch_size = tf.shape(model_input)[0]
    num_steps = tf.shape(model_input)[1]
    cells = [get_cell(i) for i in range(num_layers)]

    def body(step: int, current_input: tf.Tensor, previous_states: tf.Tensor, outputs: tf.TensorArray):
        current_states_list = []

        for i, cell in enumerate(cells):
            with tf.variable_scope("rnn_cell_%d" % i):
                cell_previous_hidden_vector = tf.squeeze(tf.slice(previous_states, [0, i, 0], [-1, 1, -1]), axis=[1])
                output, state = cell(current_input, cell_previous_hidden_vector)

                # Set current_input to output for next cell iteration
                current_input = output

                current_states_list.append(state)

        current_states = tf.concat([tf.expand_dims(state, axis=1) for state in current_states_list], axis=1)

        with tf.variable_scope("fully_connected_output"):
            final_output = tf.contrib.layers.fully_connected(current_input, num_outputs=1, activation_fn=tf.nn.sigmoid)

        outputs.write(step, final_output)

        return step + 1, current_input, current_states, outputs

    def cond(step: int, *_):
        return step < num_steps

    *_, outputs_result_ta = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=[
            0,
            tf.zeros([batch_size, layer_size]),
            tf.zeros([batch_size, num_layers, layer_size * 2]),
            tf.TensorArray(dtype=tf.float32, size=num_steps)],
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([None, layer_size]),
            tf.TensorShape([None, num_layers, layer_size * 2]),
            tf.TensorShape([])])

    return outputs_result_ta.stack()


sequence_type = types.create_from_json("""
{
    "types": [
        {
            "base": "list",
            "type": "float"
        }
    ]
}
""")[0]
