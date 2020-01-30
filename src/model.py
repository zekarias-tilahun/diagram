import tensorflow as tf
import numpy as np

IN_KEY = 'incoming'
OUT_KEY = 'outgoing'
UNF_KEY = 'unified'
TRAIN_X_KEY = 'train_x'
TRAIN_Y_KEY = 'train_y'
PRD_X_KEY = 'prd_x'
UN_INIT_STATE = None
MODEL_COMPILED_STATE = 1
FITTED_STATE = 2
EMB_PREDICTED_STATE = 3


class DiagramModel(object):

    def __init__(self, options, adj, features=None, transferred_weights=None):
        self._state = UN_INIT_STATE
        self._encoder_kernel = None
        self._embedding_kernel = None
        self._decoder_kernel = None
        self._recon_kernel = None
        self._unified_recon_kernel = None
        self._embeddings = None
        self._model = None
        self._predictor = None
        self._options = options
        self._adj = adj
        self._features = features
        self._transferred_weights = transferred_weights
        self._feed = {}
        self._build()

    def _is_compiled(self):
        return self._state >= MODEL_COMPILED_STATE

    def _is_fitted(self):
        return self._state >= FITTED_STATE

    def _is_predicted(self):
        return self._state >= EMB_PREDICTED_STATE

    def _is_ready(self):
        msg = 'The model is not initialized or has been cleaned. ' \
              'Re-initialize either by invoking the constructor or the re_build() instance method'
        assert self._state, msg
        return True

    def _has_features(self):
        return self._features is not None

    def _build(self):
        self._build_ops()
        self._build_kernels()
        self._build_inputs()
        self._build_lookups()
        self._build_loss()
        self._build_model()
        self._compile_model()

    def _build_kernels(self):
        """
        Initialize the shared layers of diagram model

        :return:
        """
        opts = self._options

        def build_kernels(layers, prefix):
            container = []
            for i in range(len(layers)):
                name = '{}_kernel_{}'.format(prefix, i + 1)
                if self._transferred_weights is not None:
                    if self._transferred_weights[prefix] is not None:
                        print('Transferring learned weights to the {}-th layer of {}'.format(i, prefix))
                        print('Weight shape: {}, bias shape: {}'.format(
                            self._transferred_weights[prefix][i][0].shape, 
                            self._transferred_weights[prefix][i][1].shape))
                        kernel = tf.keras.layers.Dense(
                            units=layers[i], activation='tanh', name=name,
                            weights=self._transferred_weights[prefix][i])
                    else:
                        kernel = tf.keras.layers.Dense(units=layers[i], activation='tanh', name=name)
                else:
                    kernel = tf.keras.layers.Dense(units=layers[i], activation='tanh', name=name)
                container.append(kernel)
            return container

        self._encoder_kernel = build_kernels(opts.layers, 'encoder')

        if self._transferred_weights is not None and self._transferred_weights['embedding'] is not None:
            print('Transferring learned weights to the embedding layer')
            self._embedding_kernel = tf.keras.layers.Dense(units=opts.d, activation='tanh', name='embedding_kernel')
        else:
            self._embedding_kernel = tf.keras.layers.Dense(units=opts.d, activation='tanh', name='embedding_kernel')

        self._decoder_kernel = build_kernels(list(reversed(opts.layers)), 'decoder')

        if self._transferred_weights is not None and self._transferred_weights['recon']['directed'] is not None:
            self._recon_kernel = tf.keras.layers.Dense(
                units=opts.number_of_nodes, activation='tanh', name='reconstruction_kernel',
                weights=self._transferred_weights['recon']['directed'])
        else:
            self._recon_kernel = tf.keras.layers.Dense(
                units=opts.number_of_nodes, activation='tanh', name='reconstruction_kernel')

        r_units = opts.number_of_nodes if self._features is None else opts.number_of_nodes + opts.number_of_features
        if self._transferred_weights is not None and self._transferred_weights['recon']['unified'] is not None:
            print('Transferring learned weights to reconstruction layer')
            self._unified_recon_kernel = tf.keras.layers.Dense(
                units=r_units, activation='tanh', name='unified_reconstruction_kernel',
                weights=self._transferred_weights['recon']['unified'])
        else:
            self._unified_recon_kernel = tf.keras.layers.Dense(
                units=r_units, activation='tanh', name='unified_reconstruction_kernel')

    def _build_ops(self):
        """
        Defines essential tensor operations to be utilized by sub classes

        :return:
        """
        opts = self._options
        if opts.unify_method == 'mean':
            self._unify_op = tf.keras.layers.Average(name='unified_input')
        elif opts.unify_method == 'sum':
            self._unify_op = tf.keras.layers.Add(name='unified_input')
        elif opts.unify_method == 'concat':
            self._concat_input = tf.keras.layers.Concatenate(name='concat_input')
            self._unify_op = tf.keras.layers.Dense(opts.layers[-1], name='unified_input')
        else:
            raise ValueError("Found unknown unifying method {}, while expecting either 'mean' or 'sum' ".format(
                opts.unify_method))

        self._dropout = None if opts.rate == 0 else tf.keras.layers.Dropout(opts.rate, name='dropout')

        # Lambda
        self._agg_op = tf.keras.layers.Lambda(
            lambda l: tf.keras.backend.mean(l, axis=1), name='aggregate_op')
        self._squeeze = tf.keras.layers.Lambda(
            lambda l: tf.keras.backend.squeeze(l, axis=1), name='squeeze_op')
        self._add_op = tf.keras.layers.Lambda(lambda l: l[0] + l[1], name='add_op')

    def _build_inputs(self):
        """
        Builds the necessary input tensor

        :return:
        """
        pass

    def _build_lookups(self):
        """
        Builds two lookup tensors for the outgoing and incoming neighborhoods of nodes

        :return:
        """
        opts = self._options
        self._incoming_adj = tf.keras.layers.Embedding(
            input_dim=opts.number_of_nodes, output_dim=opts.number_of_nodes, weights=[self._adj.T],
            trainable=False, name='incoming_adjacency')

        self._outgoing_adj = tf.keras.layers.Embedding(
            input_dim=opts.number_of_nodes, output_dim=opts.number_of_nodes, weights=[self._adj],
            trainable=False, name='outgoing_adjacency')

    def _build_loss(self):
        """
        Builds the loss function

        :return:
        """
        def reconstruction_loss(y_true, y_hat):
            weight = y_true * (self._options.beta - 1.) + 1.
            return tf.reduce_mean(tf.pow((y_true - y_hat) * weight, 2))

        self._recon_loss = reconstruction_loss

    def _build_model(self):
        """
        Constructs the entire architecture of the model

        :return:
        """
        pass

    def _compile_model(self):
        """
        Compiles the model

        :return:
        """
        pass

    def _encode(self, input_data):
        """
        Encoder layer

        :param input_data: batch input tensor
        :return:
        """
        output = input_data
        for i in range(len(self._encoder_kernel)):
            kernel = self._encoder_kernel[i]
            output = kernel(output) if self._dropout is None else self._dropout(kernel(output))

        return output

    def _embed(self, input_data):
        """
        Embedding layer

        :param input_data: batch input tensor
        :return:
        """
        return self._embedding_kernel(input_data)

    def _decode(self, input_data):
        """
        Decoder layer

        :param input_data: batch input tensor
        :return:
        """
        output = input_data
        for i in range(len(self._decoder_kernel)):
            kernel = self._decoder_kernel[i]
            output = kernel(output) if self._dropout is None else self._dropout(kernel(output))
        return output

    def _reconstruct(self, input_data, unified=False):
        """
        Reconstruction layer

        :param input_data: batch input tensor
        :param unified: if true the unified kernel, which considers undirected neighborhood
                of nodes and their content features. Otherwise reconstructs the directed neighborhood of nodes
        :return:
        """
        kernel = self._unified_recon_kernel if unified else self._recon_kernel
        return kernel(input_data)

    def _init_feed(self):
        """
        Provides training data to the model
        :return:
        """
        pass

    def train(self, epochs=1, batch_size=32):
        """
        Fits the model

        :param epochs: The number of epochs
        :param batch_size: The batch size
        :return:
        """
        self._init_feed()
        self._model.fit(x=self._feed[TRAIN_X_KEY], y=self._feed[TRAIN_Y_KEY],
                        epochs=epochs, batch_size=batch_size)
        self._state = FITTED_STATE

    def predict(self):
        """
        Predict the embeddings all nodes

        :return:
        """
        self._is_ready()
        unified_embedding, out_embedding, in_embedding = self._predictor.predict(self._feed[PRD_X_KEY])
        self._embeddings = {UNF_KEY: unified_embedding, OUT_KEY: out_embedding, IN_KEY: in_embedding}
        self._state = EMB_PREDICTED_STATE
        return self._embeddings

    def save(self, emb_path=None, weight_path=None):
        """
        Saves the model parameters

        :param emb_path: A path to node embedding file
        :param weight_path: A path to the model parameters file path
        :return:
        """
        if self._is_fitted():
            if not self._is_predicted():
                self.predict()

            if weight_path is not None:
                raise NotImplementedError()
            if emb_path is not None:
                np.savez(emb_path, unified=self._embeddings[UNF_KEY], outgoing=self._embeddings[OUT_KEY],
                         incoming=self._embeddings[IN_KEY])
        else:
            raise ValueError("The model is not trained and hence there is nothing to save")

    def load(self, emb_path=None, weight_path=None):
        """
        Loads the model parameters from the specified paths

        :param emb_path: A path to node embedding file
        :param weight_path: A path to the model parameters file path
        :return:
        """
        if weight_path is not None:
            raise NotImplementedError()
        if emb_path is not None:
            emb_files = np.load(emb_path)
            self._embeddings = {UNF_KEY: emb_files[UNF_KEY], OUT_KEY: emb_files[OUT_KEY],
                                IN_KEY: emb_files[IN_KEY]}
            return self._embeddings

    def summary(self):
        """
        Displays the model summary

        :return:
        """
        self._is_ready()
        return self._model.summary()

    def plot_model(self, path=None):
        """
        Plots the model to a path

        :param path:
        :return:
        """
        self._is_ready()
        tf.keras.utils.plot_model(self._model, 'model.png' if path is None else path)

    def clean(self):
        """
        Cleans the model

        :return:
        """
        if self._state:
            del self._model
            del self._predictor
            tf.keras.backend.clear_session()
            self._state = UN_INIT_STATE

    def re_build(self):
        """
        Rebuilds the model

        :return:
        """
        self._build()


class NodeModel(DiagramModel):

    def __init__(self, options, adj, features=None):
        super(NodeModel, self).__init__(options=options, adj=adj, features=features)
        self._node_data = np.arange(self._adj.shape[0])

    def _build_inputs(self):
        self._node = tf.keras.layers.Input(shape=(1,), name='node')

    def _build_model(self):
        node_out_adj = self._squeeze(self._outgoing_adj(self._node))
        node_in_adj = self._squeeze(self._incoming_adj(self._node))

        node_out_encoded = self._encode(node_out_adj)
        node_in_encoded = self._encode(node_in_adj)

        node = self.__unify([node_out_encoded, node_in_encoded])
        self._node_embedded = self._embed(node)

        self._node_out_embedded = self._embed(node_out_encoded)
        self._node_in_embedded = self._embed(node_in_encoded)

        node_decoded = self._decode(self._node_embedded)
        node_out_decoded = self._decode(self._node_out_embedded)
        node_in_decoded = self._decode(self._node_in_embedded)

        self._node_recon = self._reconstruct(node_decoded, unified=True)
        self._node_out_recon = self._reconstruct(node_out_decoded)
        self._node_in_recon = self._reconstruct(node_in_decoded)

    def __unify(self, inputs):
        """
        Combines the list of inputs using a pre-specified combination operation

        :param inputs: list of input tensors
        :return:
        """
        if self._options.unify_method == 'concat':
            concat_data = self._concat_input(inputs)
            unified_data = self._unify_op(concat_data)
        else:
            unified_data = self._unify_op(inputs)

        return unified_data if self._dropout is None else self._dropout(unified_data)

    def _compile_model(self):
        outputs = [self._node_recon, self._node_out_recon, self._node_in_recon]
        emb_outputs = [self._node_embedded, self._node_out_embedded, self._node_in_embedded]
        self._model = tf.keras.Model(inputs=self._node, outputs=outputs)
        self._predictor = tf.keras.Model(inputs=self._node, outputs=emb_outputs)
        self._model.compile(loss=self._recon_loss, optimizer=tf.train.AdamOptimizer(self._options.learning_rate))
        self._state = MODEL_COMPILED_STATE

    def _init_feed(self):
        self._is_ready()
        out_data = self._adj
        in_data = self._adj.T
        unified_data = out_data + in_data
        if self._has_features():
            unified_data = np.concatenate([unified_data, self._features], axis=1)
        self._feed[PRD_X_KEY] = self._feed[TRAIN_X_KEY] = self._node_data
        self._feed[TRAIN_Y_KEY] = [unified_data, out_data, in_data]
        
    def get_learned_weights(self):
        if self._is_fitted():
            return {
                'encoder': [[self._model.get_weights()[1], self._model.get_weights()[2]],
                            [self._model.get_weights()[4], self._model.get_weights()[5]]],
                'embedding': [self._model.get_weights()[6], self._model.get_weights()[7]],
                'decoder': [[self._model.get_weights()[8], self._model.get_weights()[9]],
                            [self._model.get_weights()[10], self._model.get_weights()[11]]],
                'recon': {
                    'unified': [self._model.get_weights()[12], self._model.get_weights()[13]],
                    'directed': [self._model.get_weights()[14], self._model.get_weights()[15]]
                }
            }
        print('The model parameteres are not trained. Returning NoneType')


class EdgeModel(DiagramModel):

    def __init__(self, options, adj, features=None, transferred_weights=None):
        super(EdgeModel, self).__init__(adj=adj, features=features, options=options, transferred_weights=transferred_weights)
        self._sources, self._targets = self._adj.nonzero()

    def _build_inputs(self):
        self._source = tf.keras.layers.Input(shape=(1,), name='source_input')
        self._target = tf.keras.layers.Input(shape=(1,), name='target_input')

    def _build_model(self):
        # Source data
        source_in_adj = self._squeeze(self._incoming_adj(self._source))
        source_out_adj = self._squeeze(self._outgoing_adj(self._source))
        # source_unf_adj = self._add_op([source_out_adj, source_in_adj])

        # Target data
        target_in_adj = self._squeeze(self._incoming_adj(self._target))
        target_out_adj = self._squeeze(self._outgoing_adj(self._target))

        # Encoding
        source_in_enc = self._encode(source_in_adj)
        source_out_enc = self._encode(source_out_adj)

        target_in_enc = self._encode(target_in_adj)
        target_out_enc = self._encode(target_out_adj)

        # Unify
        source_unf_enc = self._add_op([source_in_enc, source_out_enc])
        target_unf_enc = self._add_op([target_in_enc, target_out_enc])

        # Embed

        self._source_in_emb = self._embed(source_in_enc)
        self._source_out_emb = self._embed(source_out_enc)
        self._source_unf_emb = self._embed(source_unf_enc)

        target_in_emb = self._embed(target_in_enc)
        target_out_emb = self._embed(target_out_enc)
        target_unf_emb = self._embed(target_unf_enc)

        # Decode
        source_in_dec = self._decode(self._source_in_emb)
        source_out_dec = self._decode(self._source_out_emb)
        source_unf_dec = self._decode(self._source_unf_emb)

        target_in_dec = self._decode(target_in_emb)
        target_out_dec = self._decode(target_out_emb)
        target_unf_dec = self._decode(target_unf_emb)

        # Reconstruction
        self._source_in_reconstruction = self._reconstruct(source_in_dec)
        self._source_out_reconstruction = self._reconstruct(source_out_dec)
        self._source_unf_reconstruction = self._reconstruct(source_unf_dec, unified=True)

        self._target_in_reconstruction = self._reconstruct(target_in_dec)
        self._target_out_reconstruction = self._reconstruct(target_out_dec)
        self._target_unf_reconstruction = self._reconstruct(target_unf_dec, unified=True)

    def _compile_model(self):
        outputs = [self._source_in_reconstruction, self._source_out_reconstruction,
                   self._source_unf_reconstruction, self._target_in_reconstruction,
                   self._target_out_reconstruction, self._target_unf_reconstruction]
        emb_outputs = [self._source_unf_emb, self._source_out_emb, self._source_in_emb]
        self._model = tf.keras.Model(
            inputs=[self._source, self._target], outputs=outputs)
        self._predictor = tf.keras.Model(inputs=self._source, outputs=emb_outputs)
        self._model.compile(
            loss=self._recon_loss,
            optimizer=tf.train.AdamOptimizer(self._options.learning_rate))
        self._state = MODEL_COMPILED_STATE

    def _init_feed(self):
        self._is_ready()
        node_ids = np.arange(self._adj.shape[0])
        unified_data = self._adj + self._adj.T
        if self._has_features():
            unified_data = np.concatenate([unified_data, self._features], axis=1)

        s_in, s_out, s_unf = self._adj.T[self._sources], self._adj.T[self._targets], unified_data[self._sources]
        t_in, t_out, t_unf = s_out, self._adj[self._targets], unified_data[self._targets]
        self._feed[TRAIN_X_KEY] = [self._sources, self._targets]
        self._feed[TRAIN_Y_KEY] = [s_in, s_out, s_unf, t_in, t_out, t_unf]
        self._feed[PRD_X_KEY] = node_ids