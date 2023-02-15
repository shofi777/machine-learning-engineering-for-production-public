#!/usr/bin/env python

# Copyright 2019 Google Inc. All Rights Reserved. Licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# This tool does batch predictions on a trained model.

from __future__ import absolute_import
from __future__ import print_function

import argparse
import json
import os
import sys
import tempfile

import pubchem

import apache_beam as beam
import tensorflow as tf

from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.options.pipeline_options import StandardOptions
from tensorflow.python.framework import ops

class Predict(beam.DoFn):
  def __init__(self,
      model_dir,
      id_key):
    super(Predict, self).__init__()
    self.model_dir = model_dir
    self.id_key = id_key
    self.model = None

  def process(self, inputs):
    
    # Load the model only if it hasn't been loaded yet
    if not self.model:
      self.model = tf.saved_model.load(self.model_dir)

    # When using dummy data, remove predictions from inputs to
    # simulate serving data (i.e. incoming data does not contain labels) 
    if 'Energy' in inputs:
      inputs.pop('Energy')

    # Pass raw inputs to serving signature to get predictions
    results = self.model.signatures['serving_default'](**inputs)
    
    # Unpack ID key and model prediction from results tuple
    id_key = results['output_0'].numpy().tolist()
    predictions = results['output_1'].numpy().tolist()

    yield {
        'id': id_key,
        'predictions': predictions
    }


# [START dataflow_molecules_run_definition]
def run(model_dir, feature_extraction, sink, beam_options=None):
  print('Listening...')
  with beam.Pipeline(options=beam_options) as p:
    _ = (p
        | 'Feature extraction' >> feature_extraction
        | 'Predict' >> beam.ParDo(Predict(model_dir, 'ID'))
        | 'Format as JSON' >> beam.Map(json.dumps)
        | 'Write predictions' >> sink)
# [END dataflow_molecules_run_definition]


if __name__ == '__main__':
  """Main function"""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      '--work-dir',
      required=True,
      help='Directory for temporary files and preprocessed datasets to. '
           'This can be a Google Cloud Storage path.')

  parser.add_argument(
      '--model-dir',
      required=True,
      help='Path to the exported TensorFlow model. '
           'This can be a Google Cloud Storage path.')

  verbs = parser.add_subparsers(dest='verb')
  batch_verb = verbs.add_parser('batch', help='Batch prediction')
  batch_verb.add_argument(
      '--inputs-dir',
      required=True,
      help='Input directory where SDF data files are read from. '
           'This can be a Google Cloud Storage path.')
  batch_verb.add_argument(
      '--outputs-dir',
      required=True,
      help='Directory to store prediction results. '
           'This can be a Google Cloud Storage path.')

  stream_verb = verbs.add_parser('stream', help='Streaming prediction')
  stream_verb.add_argument(
      '--inputs-topic',
      default='molecules-inputs',
      help='PubSub topic to subscribe for molecules.')

  stream_verb.add_argument(
      '--outputs-topic',
      default='molecules-predictions',
      help='PubSub topic to publish predictions.')

  args, pipeline_args = parser.parse_known_args()

  beam_options = PipelineOptions(pipeline_args)
  beam_options.view_as(SetupOptions).save_main_session = True

  project = beam_options.view_as(GoogleCloudOptions).project

  # [START dataflow_molecules_batch]
  if args.verb == 'batch':
    data_files_pattern = os.path.join(args.inputs_dir, '*.sdf')
    results_prefix = os.path.join(args.outputs_dir, 'part')
    source = pubchem.ParseSDF(data_files_pattern)
    sink = beam.io.WriteToText(results_prefix)

  # [END dataflow_molecules_batch]

  else:
    parser.print_usage()
    sys.exit(1)

  # [START dataflow_molecules_call_run]
  run(
      args.model_dir,
      pubchem.SimpleFeatureExtraction(source),
      sink,
      beam_options)
  # [END dataflow_molecules_call_run]
