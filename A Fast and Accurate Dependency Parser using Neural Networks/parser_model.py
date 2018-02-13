import pickle
import os
import time
import tensorflow as tf
from general_utils import Progbar
from model import Model
from initialization import xavier_weight_init
from parser_utils import minibatches, load_and_preprocess_data


class Config(object):

    n_features = 36
    n_classes = 3
    dropout = 0.5  # (p_drop in the handout)
    embed_size = 50
    hidden_size = 200
    batch_size = 1024
    n_epochs = 2000
    lr = 0.0005


class ParserModel(Model):

    def add_placeholders(self):

        self.input_placeholder=tf.placeholder(tf.int32,[None,self.config.n_features])
        self.labels_placeholder=tf.placeholder(tf.float32,[None,self.config.n_classes])
        self.dropout_placeholder=tf.placeholder(tf.float32)


    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=0):

        feed_dict={self.input_placeholder:inputs_batch,self.dropout_placeholder:dropout}
        if labels_batch is not None :
            feed_dict[self.labels_placeholder]=labels_batch

        return feed_dict

    def add_embedding(self):

        embedded=tf.Variable(self.pretrained_embeddings)
        embeddings=tf.nn.embedding_lookup(embedded,self.input_placeholder)
        embeddings=tf.reshape(embeddings,[-1,self.config.n_features*self.config.embed_size])

        return embeddings

    def add_prediction_op(self):

        x = self.add_embedding()

        init=xavier_weight_init()
        with tf.variable_scope("transformation"):
            b1=tf.Variable(tf.zeros([self.config.hidden_size]))
            b2=tf.Variable(tf.zeros([self.config.n_classes]))
            self.W=init([self.config.n_features*self.config.embed_size,self.config.hidden_size])
            U=init([self.config.hidden_size,self.config.n_classes])
            h=tf.nn.relu(tf.matmul(x,self.W)+b1)
            h_drop=tf.nn.dropout(h,1-self.dropout_placeholder)
            pred=tf.matmul(h_drop,U)+b2

        return pred

    def add_loss_op(self, pred):

        loss=tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=self.labels_placeholder)
        loss+= 10e-6 *tf.nn.l2_loss(self.W)
        loss=tf.reduce_mean(loss)

        return loss

    def add_training_op(self, loss):


        adam_op=tf.train.AdamOptimizer()
        train_op=adam_op.minimize(loss)

        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def run_epoch(self, sess, parser, train_examples, dev_set):
        n_minibatches = 1 + len(train_examples) / self.config.batch_size
        prog = Progbar(target=n_minibatches)
        for i, (train_x, train_y) in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, train_x, train_y)
            prog.update(i + 1, [("train loss", loss)])

        print ("Evaluating on dev set")
        dev_UAS, _ = parser.parse(dev_set)
        print ("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
        return dev_UAS

    def fit(self, sess, saver, parser, train_examples, dev_set):
        best_dev_UAS = 0
        for epoch in range(self.config.n_epochs):
            print ("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
            dev_UAS = self.run_epoch(sess, parser, train_examples, dev_set)
            if dev_UAS > best_dev_UAS:
                best_dev_UAS = dev_UAS
                if saver:
                    print ("New best dev UAS! Saving model in ./data/weights/parser.weights")
                    saver.save(sess, './data/weights/parser.weights')


    def __init__(self, config, pretrained_embeddings):
        self.pretrained_embeddings = pretrained_embeddings
        self.config = config
        self.build()


def main(debug=True):
    print (80 * "=")
    print ("INITIALIZING")
    con = Config()
    print ((80 * "=") ,  con)
    parser, embeddings, train_examples, dev_set, test_set = load_and_preprocess_data(debug)
    if not os.path.exists('./data/weights/'):
        os.makedirs('./data/weights/')

    with tf.Graph().as_default() as graph:
        print ("Building model...")
        t = time.time()
        model = ParserModel(con, embeddings)
        parser.model = model
        init_op = tf.global_variables_initializer()
        saver = None if debug else tf.train.Saver()
        print ("took {:.2f} seconds\n".format(time.time() - t))
    graph.finalize()

    with tf.Session(graph=graph) as session:
        parser.session = session
        session.run(init_op)

        print (80 * "=")
        print ("TRAINING")
        print (80 * "=")
        model.fit(session, saver, parser, train_examples, dev_set)

        if not debug:
            print (80 * "=")
            print ("TESTING")
            print (80 * "=")
            print ("Restoring the best model weights found on the dev set")
            saver.restore(session, './data/weights/parser.weights')
            print ("Final evaluation on test set",)
            UAS, dependencies = parser.parse(test_set)
            print ("- test UAS: {:.2f}".format(UAS * 100.0))
            print ("Writing predictions")
            with open('q2_test.predicted.pkl', 'w') as f:
                pickle.dump(dependencies, f, -1)
            print ("Done!")


if __name__ == '__main__':
    main()

