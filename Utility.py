import json
import tensorflow as tf
# ==============================================================================
################################# AUTHOR: ######################################
############################# DIEGO PERGOLINI ##################################
# ==============================================================================

def getConfiguration(fileName):
    FLAGS = tf.flags.FLAGS
    with open(fileName) as data_file:
        configuration = json.load(data_file)
        tf.flags.DEFINE_integer("hidden_size", int(configuration["hidden_size"]), "Size of LSTM hidden layer.")
        tf.flags.DEFINE_integer("memory_size", int(configuration["memory_size"]), "The number of memory slots.")
        tf.flags.DEFINE_integer("word_size", int(configuration["word_size"]), "The width of each memory slot.")
        tf.flags.DEFINE_integer("num_write_heads", int(configuration["num_write_heads"]), "Number of memory write heads.")
        tf.flags.DEFINE_integer("num_read_heads", int(configuration["num_read_heads"]), "Number of memory read heads.")
        tf.flags.DEFINE_integer("clip_value", int(configuration["clip_value"]),
                                "Maximum absolute value of controller and dnc outputs.")


        tf.flags.DEFINE_float("max_grad_norm", float(configuration["max_grad_norm"]), "Gradient clipping norm limit.")
        tf.flags.DEFINE_float("learning_rate", float(configuration["learning_rate"]), "Optimizer learning rate.")
        tf.flags.DEFINE_float("final_learning_rate", float(configuration["final_learning_rate"]), "Optimizer final learning rate.")
        tf.flags.DEFINE_float("optimizer_epsilon", float(configuration["optimizer_epsilon"]),
                              "Epsilon used for RMSProp optimizer.")

        tf.flags.DEFINE_string("dataset",configuration["dataset"],"The dataset to use")
        tf.flags.DEFINE_string("datasetDest",configuration["datasetDest"],"The dataset  destination to use")
        tf.flags.DEFINE_string("w2v_model", configuration["w2v_model"], "The word to vec model to use")
        tf.flags.DEFINE_boolean("random", configuration["random"], "True if you want to randomized the rewiew to choose")
        tf.flags.DEFINE_boolean("seed", int(configuration["seed"]), "The seed that you want to set")

        tf.flags.DEFINE_integer("batch_size", int(configuration["batch_size"]), "Batch size for training.")
        tf.flags.DEFINE_integer("max_lenght", int(configuration["max_lenght"]), "Max number of word of the review.")
        tf.flags.DEFINE_integer("word_dimension", int(configuration["word_dimension"]), "The number of dimension of W2V")

        tf.flags.DEFINE_integer("num_classes", int(configuration["num_classes"]),
                                "Number of classes")
        tf.flags.DEFINE_integer("num_training_iterations", int(configuration["num_training_iterations"]),
                                "Number of iterations to train for.")
        tf.flags.DEFINE_integer("num_testing_iterations",int( configuration["num_testing_iterations"]),
                                "Number of iterations to train for.")
        tf.flags.DEFINE_integer("num_epochs", int(configuration["num_epochs"]),
                                "Number of epoch.")
        tf.flags.DEFINE_integer("report_interval", int(configuration["report_interval"]),
                                "Iterations between reports (samples, valid loss).")
        tf.flags.DEFINE_string("checkpoint_dir", configuration["checkpoint_dir"],
                               "Checkpointing directory.")
        tf.flags.DEFINE_integer("checkpoint_interval", int(configuration["checkpoint_interval"]),
                                "Checkpointing step interval.")

    return FLAGS,int(configuration["num_training_iterations"])+ int(configuration["num_testing_iterations"])

def memory_usage():
    """Memory usage of the current process in kilobytes."""
    status = None
    result = {'peak': 0, 'rss': 0}
    try:
        # This will only work on systems with a /proc file system
        # (like Linux).
        status = open('/proc/self/status')
        for line in status:
            parts = line.split()
            key = parts[0][2:-1].lower()
            if key in result:
                result[key] = int(parts[1])
    finally:
        if status is not None:
            status.close()
    return result
