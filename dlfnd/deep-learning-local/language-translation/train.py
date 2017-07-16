""" Language Translation Training """
import warnings
from distutils.version import LooseVersion
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import helper
import problem_unittests as tests

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""

(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""

print(LooseVersion(tf.__version__))
# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.1 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def model_inputs():
    """
    Create TF Placeholders for input, targets, learning rate, and lengths of source and target sequences.
    :return: Tuple (input, targets, learning rate, keep probability, target sequence length,
    max target sequence length, source sequence length)
    """
    # TODO: Implement Function
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='target')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
    max_target_len = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, [None], name='source_sequence_length')
                               
    return inputs, targets, learning_rate, keep_prob, target_sequence_length, \
        max_target_len, source_sequence_length

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_inputs(model_inputs)


def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for encoding
    :param target_data: Target Placehoder
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param batch_size: Batch Size
    :return: Preprocessed target data
    """
    # TODO: Implement Function
    go_id = target_vocab_to_int['<GO>']
    first_column = tf.fill([batch_size, 1], go_id)
    no_last_word = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    with_first_column = tf.concat([first_column, no_last_word], 1)
    return with_first_column


# Add a vocab id for 'GO'
max_vocab_to_int = max(target_vocab_to_int.values())
target_vocab_to_int['<GO>'] = max_vocab_to_int + 1


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_process_encoding_input(process_decoder_input)

from imp import reload
reload(tests)

def encoding_layer_1(rnn_inputs, rnn_size, num_layers, keep_prob, 
                   source_sequence_length, source_vocab_size, 
                   encoding_embedding_size):

    # Encoder embedding
    enc_embed_input = tf.contrib.layers.embed_sequence(rnn_inputs, source_vocab_size, encoding_embedding_size)

    # RNN cell
    def make_cell(rnn_size):
        enc_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return enc_cell

    enc_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])
    
    enc_output, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_embed_input, sequence_length=source_sequence_length, dtype=tf.float32)
    
    return enc_output, enc_state

def lstm_cell(rnn_size):
    return tf.contrib.rnn.BasicLSTMCell(rnn_size)

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, 
                   source_sequence_length, source_vocab_size, 
                   encoding_embedding_size):
    """
    Create encoding layer
    :param rnn_inputs: Inputs for the RNN
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param keep_prob: Dropout keep probability
    :param source_sequence_length: a list of the lengths of each sequence in the batch
    :param source_vocab_size: vocabulary size of source data
    :param encoding_embedding_size: embedding size of source data
    :return: tuple (RNN output, RNN state)
    """
    # TODO: Implement Function
    # Stack up multiple LSTM layers, for deep learning... Need to stack different multiple
    # lstm layers for tensorflow v1.2: 
    # https://stackoverflow.com/questions/44615147/valueerror-trying-to-share-variable-rnn-multi-rnn-cell-cell-0-basic-lstm-cell-k
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(rnn_size) for _ in range(num_layers)])
    enc_cell = tf.contrib.rnn.DropoutWrapper(stacked_lstm, keep_prob)

    embed = tf.contrib.layers.embed_sequence(rnn_inputs, source_vocab_size, encoding_embedding_size)
    #print(embed.get_shape())
    
    outputs, final_state = tf.nn.dynamic_rnn(enc_cell, embed, sequence_length=source_sequence_length, dtype=tf.float32)
    
    # final_state = tf.identity(final_state, name='final_state')
    return outputs, final_state

print("Testing encoding_layer")
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_encoding_layer(encoding_layer)


def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, 
                         target_sequence_length, max_summary_length, 
                         output_layer, keep_prob):
    """
    Create a decoding layer for training
    :param encoder_state: Encoder State
    :param dec_cell: Decoder RNN Cell
    :param dec_embed_input: Decoder embedded input
    :param target_sequence_length: The lengths of each sequence in the target batch
    :param max_summary_length: The length of the longest sequence in the batch
    :param output_layer: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: BasicDecoderOutput containing training logits and sample_id
    """
    # TODO: Implement Function   
    training_helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, target_sequence_length)
    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, training_helper, encoder_state, output_layer)
    final_outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=max_summary_length)
    return final_outputs


print("Testing decoding_layer_train")
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_train(decoding_layer_train)

def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_target_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob):
    """
    Create a decoding layer for inference
    :param encoder_state: Encoder state
    :param dec_cell: Decoder RNN Cell
    :param dec_embeddings: Decoder embeddings
    :param start_of_sequence_id: GO ID
    :param end_of_sequence_id: EOS Id
    :param max_target_sequence_length: Maximum length of target sequences
    :param vocab_size: Size of decoder/target vocabulary
    :param decoding_scope: TenorFlow Variable Scope for decoding
    :param output_layer: Function to apply the output layer
    :param batch_size: Batch size
    :param keep_prob: Dropout keep probability
    :return: BasicDecoderOutput containing inference logits and sample_id
    """
    # TODO: Implement Function
    start_tokens = tf.tile(tf.constant([start_of_sequence_id], dtype=tf.int32), [batch_size])
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, start_tokens, \
        end_of_sequence_id)
    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, encoder_state, output_layer)
    outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=max_target_sequence_length)
    return outputs


print("Testing decoding_layer_infer")
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_infer(decoding_layer_infer)

def decoding_layer(dec_input, encoder_state,
                   target_sequence_length, max_target_sequence_length,
                   rnn_size,
                   num_layers, target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, decoding_embedding_size):
    """
    Create decoding layer
    :param dec_input: Decoder input
    :param encoder_state: Encoder state
    :param target_sequence_length: The lengths of each sequence in the target batch
    :param max_target_sequence_length: Maximum length of target sequences
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param target_vocab_size: Size of target vocabulary
    :param batch_size: The size of the batch
    :param keep_prob: Dropout keep probability
    :param decoding_embedding_size: Decoding embedding size
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    # TODO: Implement Function
    # Embed target sequences
    embed = tf.contrib.layers.embed_sequence(dec_input, target_vocab_size, decoding_embedding_size)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    # Construct the decoder LSTM cell (just like you constructed the encoder cell above)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(rnn_size) for _ in range(num_layers)])
    dec_cell = tf.contrib.rnn.DropoutWrapper(stacked_lstm, keep_prob)

    # Create an output layer to map the outputs of the decoder to the elements of our vocabulary
    output_layer = Dense(target_vocab_size,
        kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))

    with tf.variable_scope("decoding_layer", reuse=None) as dec_scope:        
        # Use the your decoding_layer_train(encoder_state, dec_cell, dec_embed_input, target_sequence_length, 
        # max_target_sequence_length, output_layer, keep_prob) function to get the training logits.
        train_logits = decoding_layer_train(encoder_state, dec_cell, dec_embed_input, target_sequence_length, \
            max_target_sequence_length, output_layer, keep_prob)

    # Use your decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, 
    # end_of_sequence_id, max_target_sequence_length, vocab_size, output_layer, batch_size, keep_prob) 
    # function to get the inference logits
    start_id = target_vocab_to_int['<GO>']
    end_id = target_vocab_to_int['<EOS>']

    with tf.variable_scope("decoding_layer", reuse=True) as dec_scope:        
        infer_logits = decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_id, end_id, \
            max_target_sequence_length, target_vocab_size, output_layer, batch_size, keep_prob)

    return train_logits, infer_logits


print("Testing decoding_layer")
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer(decoding_layer)

def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  source_sequence_length, target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence part of the neural network
    :param input_data: Input placeholder
    :param target_data: Target placeholder
    :param keep_prob: Dropout keep probability placeholder
    :param batch_size: Batch Size
    :param source_sequence_length: Sequence Lengths of source sequences in the batch
    :param target_sequence_length: Sequence Lengths of target sequences in the batch
    :param source_vocab_size: Source vocabulary size
    :param target_vocab_size: Target vocabulary size
    :param enc_embedding_size: Decoder embedding size
    :param dec_embedding_size: Encoder embedding size
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    # TODO: Implement Function

    # Encode the input using your encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob,
    # source_sequence_length, source_vocab_size, encoding_embedding_size).
    enc_outputs, enc_state = encoding_layer(input_data, rnn_size, num_layers, keep_prob, \
        source_sequence_length, source_vocab_size, enc_embedding_size)
    
    # Process target data using your process_decoder_input(target_data, target_vocab_to_int, batch_size) \
    # function.
    dec_input = process_decoder_input(target_data, target_vocab_to_int, batch_size)
    
    # Decode the encoded input using your decoding_layer(dec_input, enc_state, target_sequence_length, \
    # max_target_sentence_length, rnn_size, num_layers, target_vocab_to_int, target_vocab_size, \
    # batch_size, keep_prob, dec_embedding_size) function.
    train_logits, infer_logits = decoding_layer(dec_input, enc_state, target_sequence_length,\
        max_target_sentence_length, rnn_size, num_layers, target_vocab_to_int, target_vocab_size, \
        batch_size, keep_prob, dec_embedding_size)
    
    return train_logits, infer_logits

print("Testing seq2seq_model")
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_seq2seq_model(seq2seq_model)

# Hyper parameters

# Number of Epochs
epochs = 15
# Batch Size
batch_size = 64
# RNN Size
rnn_size = 256
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 32 
decoding_embedding_size = 32   
# Learning Rate
learning_rate = 0.001
# Dropout Keep Probability
keep_probability = 0.5
display_step = 100

# Build the graph
import time

build_start = time.localtime()
build_start_mark = time.time()
print("Building graph started at {}".format(time.strftime("%Y-%m-%d %H:%M:%S", build_start)))
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
save_path = 'checkpoints/dev'
(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
max_target_sentence_length = max([len(sentence) for sentence in source_int_text])

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, lr, keep_prob, target_sequence_length, max_target_sequence_length, source_sequence_length = model_inputs()

    #sequence_length = tf.placeholder_with_default(max_target_sentence_length, None, name='sequence_length')
    input_shape = tf.shape(input_data)

    train_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                   targets,
                                                   keep_prob,
                                                   batch_size,
                                                   source_sequence_length,
                                                   target_sequence_length,
                                                   max_target_sequence_length,
                                                   len(source_vocab_to_int),
                                                   len(target_vocab_to_int),
                                                   encoding_embedding_size,
                                                   decoding_embedding_size,
                                                   rnn_size,
                                                   num_layers,
                                                   target_vocab_to_int)


    training_logits = tf.identity(train_logits.rnn_output, name='logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

build_end = time.localtime()
build_end_mark = time.time()
build_duration = build_end_mark - build_start_mark

print("Building graph ended at {} and took {:.2f} seconds".format(time.strftime("%Y-%m-%d %H:%M:%S", build_end), build_duration))

# Batch and patch the source and target sequences
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths

# Training
train_start = time.localtime()
train_start_mark = time.time()
print("Training started at {}".format(time.strftime("%Y-%m-%d %H:%M:%S", train_start)))
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))

# Split data to training and validation sets
train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]
valid_source = source_int_text[:batch_size]
valid_target = target_int_text[:batch_size]
(valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths ) = next(get_batches(valid_source,
                                                                                                             valid_target,
                                                                                                             batch_size,
                                                                                                             source_vocab_to_int['<PAD>'],
                                                                                                             target_vocab_to_int['<PAD>']))                                                                                                  
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                get_batches(train_source, train_target, batch_size,
                            source_vocab_to_int['<PAD>'],
                            target_vocab_to_int['<PAD>'])):

            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 source_sequence_length: sources_lengths,
                 keep_prob: keep_probability})


            if batch_i % display_step == 0 and batch_i > 0:


                batch_train_logits = sess.run(
                    inference_logits,
                    {input_data: source_batch,
                     source_sequence_length: sources_lengths,
                     target_sequence_length: targets_lengths,
                     keep_prob: 1.0})


                batch_valid_logits = sess.run(
                    inference_logits,
                    {input_data: valid_sources_batch,
                     source_sequence_length: valid_sources_lengths,
                     target_sequence_length: valid_targets_lengths,
                     keep_prob: 1.0})

                train_acc = get_accuracy(target_batch, batch_train_logits)

                valid_acc = get_accuracy(valid_targets_batch, batch_valid_logits)

                print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                      .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')

# Save Parameters
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params(save_path)

train_end = time.localtime()
train_end_mark = time.time()
train_duration = train_end_mark - train_start_mark
print("Training ended at {} and took {:.2f} seconds".format(time.strftime("%Y-%m-%d %H:%M:%S", train_end), train_duration))
