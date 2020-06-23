## Utils.py -- Some utility functions
##
## Copyright (C) 2018, IBM Corp
##                     Chun-Chen Tu <timtu@umich.edu>
##                     PaiShun Ting <paishun@umich.edu>
##                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

from PIL import Image

import os
import numpy as np


def load_AE(codec_prefix, print_summary=False):
    save_file_prefix = codec_prefix + "_"

    decoder_model_filename = save_file_prefix + "decoder.json"
    decoder_weight_filename = save_file_prefix + "decoder.h5"

    if not os.path.isfile(decoder_model_filename):
        raise Exception("The file for decoder model does not exist:{}".format(decoder_model_filename))
    json_file = open(decoder_model_filename, 'r')
    decoder = model_from_json(json_file.read(), custom_objects={"tf": tf})
    json_file.close()

    if not os.path.isfile(decoder_weight_filename):
        raise Exception("The file for decoder weights does not exist:{}".format(decoder_weight_filename))
    decoder.load_weights(decoder_weight_filename)

    if print_summary:
        print("Decoder summaries")
        decoder.summary()

    return decoder


def load_codec(codec_prefix, print_summary=False):
    # load data
    saveFilePrefix = codec_prefix + '_'
    # load models
    encoder_model_filename = saveFilePrefix + "encoder.json"
    decoder_model_filename = saveFilePrefix + "decoder.json"
    encoder_weight_filename = saveFilePrefix + "encoder.h5"
    decoder_weight_filename = saveFilePrefix + "decoder.h5"

    if not os.path.isfile(encoder_model_filename):
        raise Exception("The file for encoder model does not exist:{}".format(encoder_model_filename))

    json_file = open(encoder_model_filename, 'r')
    encoder = model_from_json(json_file.read(), custom_objects={"tf": tf})
    json_file.close()

    if not os.path.isfile(encoder_weight_filename):
        raise Exception("The file for encoder weights does not exist:{}".format(encoder_weight_filename))
    encoder.load_weights(encoder_weight_filename)

    if not os.path.isfile(decoder_model_filename):
        raise Exception("The file for decoder model does not exist:{}".format(decoder_model_filename))
    json_file = open(decoder_model_filename, 'r')
    decoder_temp = model_from_json(json_file.read(), custom_objects={"tf": tf})
    json_file.close()

    if not os.path.isfile(decoder_weight_filename):
        raise Exception("The file for decoder weights does not exist:{}".format(decoder_weight_filename))
    decoder_temp.load_weights(decoder_weight_filename)

    if print_summary:
        print("Encoder summaries")
        encoder.summary()

    _, encode_H, encode_W, numChannels = encoder.output_shape

    # the workaround
    # use configures to construct the decoder model
    # and then load the weights for each layer
    # Note that the information in configures[0::] is the sequential model for encoder
    # thus, we need to exclude the first element

    config = decoder_temp.get_config()
    config2 = config[1::]
    config2[0]['config']['batch_input_shape'] = (None, encode_H, encode_W, numChannels)
    decoder = Sequential.from_config(config2, custom_objects={"tf": tf})

    # set weights
    cnt = -1
    for l in decoder_temp.layers:
        cnt += 1
        if cnt == 0:
            continue
        weights = l.get_weights()
        decoder.layers[cnt - 1].set_weights(weights)
    if print_summary:
        print("Decoder summaries")
        decoder.summary()

    return encoder, decoder


def save_img(img, name="output.png"):
    np.save(name, img)
    fig = np.around((img + 0.5) * 255)
    fig = fig.astype(np.uint8).squeeze()
    pic = Image.fromarray(fig)
    pic.save(name)


def generate_data(data, id, target_label):
    inputs = []
    target_vec = []

    inputs.append(data.test_data[id])
    target_vec.append(np.eye(data.test_labels.shape[1])[target_label])

    inputs = np.array(inputs)
    target_vec = np.array(target_vec)

    return inputs, target_vec


def generate_attack_data_set(data, num_sample, img_offset, model, attack_type="targeted", random_target_class=None,
                             shift_index=False):
    """
    Generate the data for conducting attack. Only select the data being classified correctly.
    """
    orig_img = []
    orig_labels = []
    target_labels = []
    orig_img_id = []

    pred_labels = np.argmax(model.target_model.predict(data.test_data), axis=1)
    true_labels = np.argmax(data.test_labels, axis=1)
    correct_data_indices = np.where([1 if x == y else 0 for (x, y) in zip(pred_labels, true_labels)])

    print(
        "Total testing data:{}, correct classified data:{}".format(len(data.test_labels), len(correct_data_indices[0])))

    data.test_data = data.test_data[correct_data_indices]
    data.test_labels = data.test_labels[correct_data_indices]
    true_labels = true_labels[correct_data_indices]

    np.random.seed(img_offset)  # for parallel running
    class_num = data.test_labels.shape[1]
    for sample_index in range(num_sample):

        if attack_type == "targeted":
            if random_target_class is not None:
                # randomly select one class to attack, except the true labels
                print(random_target_class)
                seq = np.random.choice(random_target_class, 1)
                while seq == true_labels[img_offset + sample_index]:
                    seq = np.random.choice(random_target_class, 1)

            else:
                seq = list(range(class_num))
                seq.remove(true_labels[img_offset + sample_index])

            for s in seq:
                if shift_index and s == 0:
                    s += 1
                orig_img.append(data.test_data[img_offset + sample_index])
                target_labels.append(np.eye(class_num)[s])
                orig_labels.append(data.test_labels[img_offset + sample_index])
                orig_img_id.append(img_offset + sample_index)

        elif attack_type == "untargeted":
            orig_img.append(data.test_data[img_offset + sample_index])
            target_labels.append(data.test_labels[img_offset + sample_index])
            orig_labels.append(data.test_labels[img_offset + sample_index])
            orig_img_id.append(img_offset + sample_index)

    orig_img = np.array(orig_img)
    target_labels = np.array(target_labels)
    orig_labels = np.array(orig_labels)
    orig_img_id = np.array(orig_img_id)

    return orig_img, target_labels, orig_labels, orig_img_id


def model_prediction(model, inputs):
    prob = model.target_model.predict(inputs)
    predicted_class = np.argmax(prob)
    prob_str = np.array2string(prob).replace('\n', '')
    return prob, predicted_class, prob_str
