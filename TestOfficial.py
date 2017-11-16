# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================================================================
################################# AUTHOR: ######################################
############################# DIEGO PERGOLINI ##################################
# ==============================================================================
"""Script to train the DNC on a sentiment classification task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import json
import tensorflow as tf
from Utility import getConfiguration
from Utility import  memory_usage
import dnc
import argparse
import math
import time

parser = argparse.ArgumentParser()
parser.add_argument("--configuration", help="The configuration file")
args = parser.parse_args()
if args.configuration:
    print(args.configuration)
    FLAGS,numLines = getConfiguration(args.configuration)
else:
    FLAGS, numLines = getConfiguration("configuration.json")


if FLAGS.num_classes == 5:
    from ReviewReaderWithTitleFined import balancedGetDataset
    from ReviewReaderWithTitleFined import getBalancedReviewsAndOverall
elif FLAGS.num_classes == 2:
    from ReviewReaderGensim import balancedGetDataset
    from ReviewReaderGensim import getBalancedReviewsAndOverall

def run_model(input_sequence, output_size):
  """Runs model on input sequence."""

  access_config = {
      "memory_size": FLAGS.memory_size,
      "word_size": FLAGS.word_size,
      "num_reads": FLAGS.num_read_heads,
      "num_writes": FLAGS.num_write_heads,
  }
  controller_config = {
      "hidden_size": FLAGS.hidden_size,
  }
  clip_value = FLAGS.clip_value
  #Creo la cella dnc
  dnc_core = dnc.DNC(access_config, controller_config, output_size, clip_value)
  initial_state = dnc_core.initial_state(FLAGS.batch_size)

  #Funzione che ritorna una coppia (output,state), dove output in questo caso sara un tensore
  #di forma  [batch_size,max_time,cell.output_size] perche il flag time_major e impostato a False
  #se lo si setta a True invece la forma dell'output diventa [max_time,batch_size,cell.output_size].
  output_sequence, _ = tf.nn.dynamic_rnn(
      cell=dnc_core,
      inputs=input_sequence,
      time_major=False,
      initial_state=initial_state)

  return output_sequence

"""Addestra la DNC ed al termine di ogni epoca di addestramento viene effettuata una fase di test"""
def execute(dictSize,num_training_iterations, report_interval):
  max_lenght = FLAGS.max_lenght
    #Placeholder che andra' a contenere il batch di label relativa alle recensioni
  y_= tf.placeholder(tf.float32,shape=[FLAGS.batch_size,max_lenght,FLAGS.num_classes])

  #Placeholder che andra' a contenere il batch di recensioni opportunamente codificate
  x = tf.placeholder(tf.float32, [FLAGS.batch_size, max_lenght, dictSize])

  #Placeholder che andra' a contenere il batch di maschere da applicare per il calcolo della cross-entropy
  mask = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,max_lenght])

  #Richiamando il metodo run_model ottengo la sequenza prodotta dalla rete
  output_logits = run_model(x, FLAGS.num_classes)

  #Calcolo della cross entropy totale tra le labels e gli output prodotti dalla rete
  cross = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output_logits)

  #Calcolo l'errore relativo ai singoli batch, applicando una maschera che considera gli output prodotti dalla rete
  #solo negli unrolling corrispondenti a delle parole della recensione e non considerando quelli prodotti in corrispondenza
  #di padding. La maschera applicata fornisce peso via via crescente mano a mano che si procede verso le ultime parole
  #della recensione.
  batch_error = tf.reduce_sum(cross * mask, 1)

  #Faccio la media degli errori dei singoli batch
  total_error = tf.reduce_mean(batch_error)

  #Ricavo la polarita' che la rete ha indicato all'ultima parola di ogni recensione,
  prediction = tf.arg_max(output_logits[:,max_lenght-1], 1)

  #Ricavo la polarita' indicata dalle label
  expected = tf.arg_max(y_[:,max_lenght-1], 1)

  #Ricavo quante predizioni della polarita' sono state fatte correttamente
  correct_prediction = tf.equal(prediction, expected)

  #Ricavo cosi' l'accuratezza ottenuta in questo batch
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  #Set up optimizer with global norm clipping.
  trainable_variables = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(
      tf.gradients(total_error, trainable_variables), FLAGS.max_grad_norm)

  global_step = tf.get_variable(
      name="global_step",
      shape=[],
      dtype=tf.int64,
      initializer=tf.random_uniform_initializer(-1, 1),
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

  #Placeholder che conterra' il learning rate
  learning_rate = tf.placeholder(tf.float32)

  optimizer = tf.train.RMSPropOptimizer(
      learning_rate, epsilon=FLAGS.optimizer_epsilon)

  #Passo di addestramento da eseguire per addestrare la rete
  train_step = optimizer.apply_gradients(
      zip(grads, trainable_variables), global_step=global_step)

  #Oggetto per salvare lo stato della rete.
  saver = tf.train.Saver()

  #Impostazione dei parametri relativi al salvataggio dello stato della rete.
  if FLAGS.checkpoint_interval > 0:
    hooks = [
        tf.train.CheckpointSaverHook(
            checkpoint_dir=FLAGS.checkpoint_dir,
            save_steps=FLAGS.checkpoint_interval,
            saver=saver)
    ]
  else:
    hooks = []

  #Viene scritto un file di log dell'esecuzione corrente.
  date = time.strftime("%b%d%H:%M:%S")
  outputFile = open("Experiment"+date+".txt",'w')

  with open(args.configuration) as data_file:
    configuration = json.load(data_file)
    outputFile.write(json.dumps(configuration))

  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.4
  #Viene inizializzata una sessione
  with tf.train.SingularMonitoredSession(
      hooks=hooks, checkpoint_dir=FLAGS.checkpoint_dir,config=config) as sess:

        #Il numero di istanze di training  e testing viene diviso per due perche' indica che il numero di recensioni
        # positive e negative siano uguali


        if FLAGS.num_classes ==5:
            numTraining = (FLAGS.num_training_iterations)
            numTesting = FLAGS.num_testing_iterations
            allLines = getBalancedReviewsAndOverall(numTraining,numTesting,FLAGS.dataset, FLAGS.max_lenght, FLAGS.random, FLAGS.seed)
        elif FLAGS.num_classes ==2:
            numTraining = (FLAGS.num_training_iterations // 2)
            numTesting = (FLAGS.num_testing_iterations // 2)
            #Vengono ottenute tutte le recensioni necessarie per training e testing
            allLines = getBalancedReviewsAndOverall(numTraining,numTraining,
                                                numTesting,numTesting,
                                                FLAGS.dataset,FLAGS.max_lenght,FLAGS.random,FLAGS.seed)

        #Viene inizializzato un generatore attraverso il quale ottenere mano a mano i vari batch di training e testing,
        # viene inoltre indicato quale modello word2vec debba essere utilizzato
        generator = balancedGetDataset(allLines,FLAGS.w2v_model,FLAGS)

        randomizer = random.Random(1)



        best_test_accuracy = 0


        date = time.strftime("%H:%M:%S")

#################TRAINING########################################################
        # Testing
        test_accuracy = 0
        total_test_accuracy = 0

        #################################################TESTING########################################################
        for test_iteration in range(0,(FLAGS.num_testing_iterations//FLAGS.batch_size)):

            datasetTest = next(generator)
            act_accuracy = sess.run(accuracy,
                                    {x: datasetTest[0],
                                     y_: datasetTest[1],
                                     mask: datasetTest[2]})
            test_accuracy += act_accuracy
            total_test_accuracy += act_accuracy
            if (test_iteration + 1) % report_interval == 0:

                tf.logging.info("%d: Avg testing accuracy %f.\n",
                                test_iteration+1, test_accuracy / report_interval
                                )
                info4 = str(test_iteration+1)+ ": Avg testing accuracy: "+str(test_accuracy / report_interval)+"\n"
                outputFile.write(info4)
                test_accuracy = 0
        tf.logging.info("Epoch: %d,Iteration: %d, Average Testing accuracy: %f\n",1, test_iteration+1,
                        total_test_accuracy /( FLAGS.num_testing_iterations//FLAGS.batch_size))
        info5 = "Epoch: "+str(1)+",Iteration: "+ str(test_iteration+1)+", Average Testing accuracy: "+str(total_test_accuracy /( FLAGS.num_testing_iterations//FLAGS.batch_size))+"\n"
        outputFile.write(info5)

        epoch_test_accuracy = total_test_accuracy /( FLAGS.num_testing_iterations//FLAGS.batch_size)
        if epoch_test_accuracy > best_test_accuracy:
            best_test_accuracy = epoch_test_accuracy

  date = time.strftime("%H:%M:%S")
  outputFile.close()

def main(unused_argv):
  tf.logging.set_verbosity(3)  # Print INFO log messages.

  execute(FLAGS.word_dimension,FLAGS.num_training_iterations, FLAGS.report_interval)


if __name__ == "__main__":
  tf.app.run()
