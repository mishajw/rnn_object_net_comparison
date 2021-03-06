import numpy as np
import tensorflow as tf
import tf_utils


def train(args, data: np.array, layer_size: int, num_layers: int):
    data_holder = tf_utils.data_holder.DataHolder(args, lambda i: (data[i], None), len(data))
    data_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 50])
    predictions = __get_rnn_model(layer_size, num_layers, data_placeholder)
    cost = tf.sqrt(tf.reduce_sum(tf.squared_difference(predictions, data_placeholder)))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    summary = tf.summary.scalar("cost", cost)

    # Run training
    def train_step(session, step, training_input, _, summary_writer):
        training_input, _ = training_input
        _, cost_result, summaries = session.run(
            [optimizer, cost, summary],
            {data_placeholder: training_input})

        summary_writer.add_summary(summaries, step)

    def test_step(session, step, testing_input, _, summary_writer):
        testing_input, _ = testing_input
        predictions_result, cost_result, summaries = session.run(
            [predictions, cost, summary],
            {data_placeholder: testing_input})

        summary_writer.add_summary(summaries, step)

        print("Testing cost: %f" % cost_result)

    runner = tf_utils.generic_runner.GenericRunner.from_args(args, "rnn_object_net_comparison")
    runner.set_data_holder(data_holder)
    runner.set_train_step(train_step)
    runner.set_test_step(test_step)
    runner.run()


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
            final_output = tf.squeeze(
                tf.contrib.layers.fully_connected(current_input, num_outputs=1, activation_fn=tf.nn.sigmoid), axis=1)

        outputs = outputs.write(step, final_output)

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

    # Transpose the outputs to make batch the first dimension
    return tf.transpose(outputs_result_ta.stack(), [1, 0])
