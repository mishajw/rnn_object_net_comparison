from object_net import object_net_components
from object_net import object_net_writer
from object_net import padder
from object_net import types
import argparse
import tensorflow as tf
import tf_utils


def main():
    parser = argparse.ArgumentParser()
    tf_utils.generic_runner.add_arguments(parser)
    tf_utils.data_holder.add_arguments(parser)
    args = parser.parse_args()

    data = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]] * 100
    data = map(sequence_type.get_state_output_pairs, data)
    data = padder.PaddedData.from_unpadded(data)
    data_placeholder = padder.PlaceholderPaddedData()
    data_holder = tf_utils.data_holder.DataHolder(args, lambda i: data[i], len(data))

    def get_model(training: bool) -> object_net_writer.ObjectNetWriter:
        return object_net_writer.ObjectNetWriter(
            truth_padded_data=data_placeholder,
            initial_hidden_vector_input=tf.zeros(args.batch_size),
            hidden_vector_size=128,
            object_type=sequence_type,
            training=training,
            hidden_vector_network=object_net_components.LstmHiddenVectorNetwork(
                128, 2, object_net_components.AdditionHiddenVectorCombiner()))

    model = get_model(True)
    test_model = get_model(False)

    optimizer = tf.train.AdamOptimizer().minimize(model.cost)
    tf.summary.scalar("object_net_cost", model.cost)

    # Run training
    def train_step(session, step, training_input, _, summary_writer, all_summaries):
        result, cost, summaries = session.run(
            [optimizer, model.cost, all_summaries],
            data_placeholder.get_feed_dict(training_input))

        summary_writer.add_summary(summaries, step)

        print("Training error: %s" % cost)

    def test_step(session, step, testing_input, _, summary_writer, all_summaries):
        cost_result, all_summaries = session.run(
            [model.cost, all_summaries],
            data_placeholder.get_feed_dict(testing_input))

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
                    test_model.generated_states_padded,
                    test_model.generated_outputs_padded,
                    test_model.generated_outputs_counts_padded,
                    test_model.generated_step_counts],
                data_placeholder.get_feed_dict(model_input))

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


if __name__ == "__main__":
    main()
