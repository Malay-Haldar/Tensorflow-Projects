"""Create a range of logistic regression problems with varying number of parameters.
Use Tensorflow to train models to solve the logistic regression. Plot the relationship
between number of training samples and fscore of the trained models.
For a introduction to logistic regression, please refer to
  https://www.youtube.com/watch?v=tEk6ikTKGYU
"""
import tensorflow as tf
import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt

# The maximum number of model parameters to explore.
kMaxNumParameters = 256
# The maximum multiple of model parameters to use as number of training samples.
kMaxTrainingSamplesMultiple = 16
kMaxNumTrainingSamples = kMaxNumParameters * kMaxTrainingSamplesMultiple
# The threshold to use for classification
kLogisticClassificationThreshold = 0.5
# The regularization parameter (lambda) of cost function.
# Please refer to https://www.youtube.com/watch?v=ef2OPmANLaM 
kRegularizationParameter = 0.1
# The number of explorations for a given number of model parameters. Increase this
# to smooth out the curve for a particular number of model parameters.
kNumModelIterations = 10
# Random data representing the input feature space.
xref_data = np.float32(np.random.uniform(-1, 1, [kMaxNumParameters, kMaxNumTrainingSamples]))
# Random data representing the test samples.
tref_data = np.float32(np.random.uniform(-1, 1, [kMaxNumParameters, 1000]))
# The ratio of training labels that are noisy, i.e, opposite of what they
# really should be. 0.15 indicated 15% of the training labels are noisy.
kNoisyLabelsRatio = 0.15
# Noise to be added to the training labels.
noise = np.random.uniform(0.0, 1.0, [1, kMaxNumTrainingSamples]) < kNoisyLabelsRatio
print "Noise = " , np.float32(np.sum(noise)) / kMaxNumTrainingSamples, "%"

def LogisticClassification(model, data):
  """
  Apply logistic classfication to the data according to:
    y = 1 if sigmoid(model.data) > threshold, 0 otherwise
    Reference: https://www.youtube.com/watch?v=tEk6ikTKGYU
  Args:
    model: The parameters of the logistic regression model.
    data: The instance to be classified.
  Returns:
    Classification result. 
  """
  return 1 / (1 + np.exp(-np.dot(model, data))) > kLogisticClassificationThreshold

def GetFscore(truth, observed):
  """
  Computes f-score based on the differences between supplied truth and observed.
  Reference: https://en.wikipedia.org/wiki/Precision_and_recall
  Args:
    truth: The ground truth against which the f-score is computed.
    observed: The observations for which the f-score is computed.
  Returns:
    The computed f-score.
  """
  # True positives.
  tp = np.sum(np.logical_and(truth, observed))
  # False positives.
  fp = np.sum(np.logical_and(np.logical_not(truth), observed))
  # False negatives.
  fn = np.sum(np.logical_and(truth, np.logical_not(observed)))
  # True negatives.
  tn = np.sum(np.logical_and(np.logical_not(truth), np.logical_not(observed)))
  precision = np.float32(tp) / (tp + fp)
  recall = np.float32(tp) / (tp + fn)
  fscore = 2.0 * precision * recall / (precision + recall) 
  return fscore
  
def RunModel(num_parameters, num_training_samples):
  """
  Creates a logistic regression model of size num_parameters by initializing
  the parameters randomly. Generates ground truth for the model and adds
  the specified amount of noise to the labels. Then uses gradient decent to
  train the model and determines the f-score of the trained model on a 
  randomly generated test.
  Args:
    num_parameters: The number of parameters of the logistic regression model.
    num_training_samples The number of training samples used to train the model.
  Returns:
    The f-score of the trained model. 
  """

  # Generate a model by randomly choosing it's parameters.
  model = np.random.uniform(-1, 1, [1, num_parameters]) 
  # Choose the subset of the total feature space applicable for this model.
  x_data = xref_data[0:num_parameters, 0:num_training_samples]
  # Apply the model on the input features. 
  y_labels_no_noise = LogisticClassification(model, x_data)
  # Generate the training labels by adding some noise to the model output.
  # The xor function flips a fraction of the labels given.
  y_labels = np.logical_xor(y_labels_no_noise, noise[0, 0:num_training_samples])

                              
  # Construct a logistic regression model. Alternatively, one can also use
  # tf.nn.softmax_cross_entropy_with_logits() for compactness. This version
  # makes the model explicit by exposing the basic units.
  W = tf.Variable(tf.random_uniform([1, num_parameters], -1.0, 1.0))
  y = tf.sigmoid(tf.matmul(W, x_data)) 
  # Contruct the regularization term.
  r = tf.mul(kRegularizationParameter, tf.reduce_sum(tf.square(W)))

  # Mimize the L2 loss against the given labels along with square of
  # the parameters for regularization.
  loss = tf.add(tf.nn.l2_loss(tf.sub(y, y_labels)), r)
  optimizer = tf.train.GradientDescentOptimizer(0.5)
  train = optimizer.minimize(loss)

  # For initializing the variables.
  init = tf.initialize_all_variables()

  # Launch the Tensorflow graph
  sess = tf.Session()
  sess.run(init)

  # Fit the plane.
  for step in xrange(0, 1000):
    sess.run(train)
  
  predicted_model = sess.run(W)

  # The subset of the test data used for this model.
  t_data = tref_data[0:num_parameters, :]
  # The computed test labels.
  yt_data = LogisticClassification(model, t_data)
  # The labels predicated by the model for the test data.
  predicted_data = LogisticClassification(predicted_model, t_data)
  # Return peformance of the model.  
  return GetFscore(yt_data, predicted_data)

# Generate model with number of parameters as 10, 40, 70,..
N = kMaxNumParameters
plots = []
legends = []
for n in [N/8, N/4, N/2, N]:
  if (n < 4):
    continue
  # Collect the performance data for the plot
  data = []
  # Vary the samples as 1x, 2x, 3x.. of parameters
  for samples in range(n, kMaxTrainingSamplesMultiple * n, n):
    score_sum = 0.0
    # Number of times each model is repeated.
    for iter in range(0, kNumModelIterations):
      score = RunModel(n, samples)
      print "Params=", n, " Samples=", samples, " Iter=", iter, " Score=", score
      score_sum += score
    # Average the score over multiple iterations to smooth out the curves
    data.append(score_sum / kNumModelIterations)
  p, = plt.plot(data, label=str(n))
  plots.append(p)
  legends.append(str(n))
plt.legend(plots, legends)
plt.ylabel("f-score")
plt.xlabel("num-training-samples / num-model-parameters")
plt.show()  
