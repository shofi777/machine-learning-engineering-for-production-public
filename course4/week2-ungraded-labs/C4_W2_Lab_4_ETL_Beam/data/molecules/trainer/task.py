from typing import List, Text

import argparse
import dill as pickle

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils

# Unpack the contents of the constants module
_NUMERIC_FEATURE_KEYS = ['NormalizedC', 'NormalizedH', 'NormalizedN', 'NormalizedO'] 

def _reader_fn(filenames):
  '''Load compressed dataset
  
  Args:
    filenames - filenames of TFRecords to load

  Returns:
    TFRecordDataset loaded from the filenames
  '''

  # Load the dataset.
  return tf.data.TFRecordDataset(filenames)


def _input_fn(file_pattern,
              tf_transform_output,
              label_key=None,
              num_epochs=None,
              batch_size=64) -> tf.data.Dataset:
  '''Create batches of features and labels from TF Records

  Args:
    file_pattern - List of files or patterns of file paths containing Example records.
    tf_transform_output - transform output graph
    num_epochs - Integer specifying the number of times to read through the dataset. 
            If None, cycles through the dataset forever.
    label_key - name of the label column
    batch_size - An int representing the number of records to combine in a single batch.

  Returns:
    A dataset of dict elements, (or a tuple of dict elements and label). 
    Each dict maps feature keys to Tensor or SparseTensor objects.
  '''

  # Get post-transfrom feature spec
  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())
  
  # Create batches of data
  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=transformed_feature_spec,
      reader=_reader_fn,
      num_epochs=num_epochs,
      label_key=label_key
      )
  
  return dataset


def _get_serve_tf_examples_fn(model, tf_transform_output, feature_spec):
  """Returns a function that applies data transformation and generates predictions"""

  # Get transformation graph
  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(inputs_list):
    """Returns the output to be used in the serving signature."""
    
    # Create a shallow copy of the dictionary in the single element list
    inputs = inputs_list[0].copy()

    # Pop ID since it is not needed in the transformation graph
    # Also needed to identify predictions
    id_key = inputs.pop('ID')
    
    # Apply data transformation to the raw inputs
    transformed = model.tft_layer(inputs)

    # Pass the transformed data to the model to get predictions
    predictions = model(transformed.values())

    return id_key, predictions

  return serve_tf_examples_fn


def _build_keras_model(hidden_units: List[int] = None) -> tf.keras.Model:
  """Creates a DNN Keras model for the regression problem.

  Args:
    hidden_units: [int], the layer sizes of the DNN (input layer first).

  Returns:
    A keras Model.
  """

  # Use helper function to create the model
  model = dnn_regressor(
      dnn_hidden_units=hidden_units or [100, 70, 50, 25])
  
  return model


def dnn_regressor(dnn_hidden_units):
  """Build a dense neural network using the Functional API.

  Args:
    dnn_hidden_units: [int], the layer sizes of the hidden DNN.

  Returns:
    A Keras model
  """

  # Define input layers for numeric keys
  input_numeric = [
      tf.keras.layers.Input(name=colname, shape=(1,), dtype=tf.float32)
      for colname in _NUMERIC_FEATURE_KEYS
  ]

  # Concatenate numeric inputs
  deep = tf.keras.layers.concatenate(input_numeric)

  # Create dense network for numeric inputs
  for numnodes in dnn_hidden_units:
    deep = tf.keras.layers.Dense(numnodes)(deep)
                                              
  # Define output of the regression model
  output = tf.keras.layers.Dense(
      1)(deep)

  # Create the Keras model
  model = tf.keras.Model(input_numeric, output)

  # Define training parameters
  model.compile(
      loss='mse',
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
      metrics='accuracy')
  
  # Print model summary
  model.summary()

  return model


def run_fn(fn_args):
  """Defines and trains the model.
  
  Args:
    fn_args: Holds training parameters as name/value pairs.
        - input_feature_spec - spec describing the raw inputs
        - train_files_pattern - file pattern of the train set files
        - eval_files_pattern - file pattern of the eval set files
        - labels - label column
        - transform_output - root directory of the transform graph and metadata
        - serving_model_dir - output directory of the trained model
  """

  # Number of nodes in the first layer of the DNN
  first_dnn_layer_size = 128
  num_dnn_layers = 2
  dnn_decay_factor = 0.5

  # Get transform output (i.e. transform graph) wrapper
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  # Load label
  label_key = fn_args.labels[0]

  # Create batches of train and eval sets
  train_dataset = _input_fn(fn_args.train_files_pattern, tf_transform_output, label_key, 10)
  eval_dataset = _input_fn(fn_args.eval_files_pattern, tf_transform_output, label_key, 10)

  # Build the model
  model = _build_keras_model(
      # Construct layers sizes with exponential decay
      hidden_units=[
          max(2, int(first_dnn_layer_size * dnn_decay_factor**i))
          for i in range(num_dnn_layers)
      ])


  # Train the model
  model.fit(
      train_dataset,
      steps_per_epoch=100,
      validation_data=eval_dataset
      )
  
  # Define input signature as dictionary of tensors
  signatures_dict = {
    'TotalC': tf.TensorSpec(shape=[None], dtype=tf.int64, name='TotalC'),
    'TotalH': tf.TensorSpec(shape=[None], dtype=tf.int64, name='TotalH'),
    'TotalO': tf.TensorSpec(shape=[None], dtype=tf.int64, name='TotalO'),
    'TotalN': tf.TensorSpec(shape=[None], dtype=tf.int64, name='TotalN'),
    'ID': tf.TensorSpec(shape=[None], dtype=tf.int64, name='ID')
    }

  # Load input feature spec
  input_feature_spec = fn_args.input_feature_spec

  # Define default serving signature
  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    tf_transform_output, input_feature_spec).get_concrete_function(
                                        [signatures_dict])
  }
  

  # Save model with signature
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures, include_optimizer=False)

# Helper function to load serialized Python objects
def load(filename):
  with tf.io.gfile.GFile(filename, 'rb') as f:
    return pickle.load(f)

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      '--work-dir',
      required=True,
      help='Directory for staging and working files. '
           'This can be a Google Cloud Storage path.')

  args = parser.parse_args()

  # Load Python objects from Preprocessing phase
  preprocess_data = load('/content/results/PreprocessData')
  
  # Assign as parameters to the task runner
  fn_args = preprocess_data

  # Add the working directory to find the transformation graph
  fn_args.transform_output = args.work_dir

  # Add the output directory for the trained model
  fn_args.serving_model_dir = f'{args.work_dir}/model'

  # Start the task runner
  run_fn(fn_args)