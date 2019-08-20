"""
    Main workflow for inference
"""
import os
import argparse
import torch
import time
import urllib
import zipfile
from pytorch_pretrained_bert import BertConfig

from prepro.process import processor
from models.model_build import Summarizer
from models.model_train import Trainer

LINK = {'stanford-corenlp': 'http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip'}


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_input():

    print("[INFO] Starting BERTSUM ...")
    # Receive raw input from user
    raw_input = input("[INFO] Enter the text:\n\n")

    # Process the raw input for inference
    example = processor(raw_input)
    return example


def configure_tokenizer(args):

    def download_and_extract(link, dir):
        print("Downloading and extracting {}...".format(link))
        data_file = "{}.zip".format(link)
        urllib.request.urlretrieve(LINK[link], data_file)
        with zipfile.ZipFile(data_file) as zip_ref:
            zip_ref.extractall(dir)
        os.remove(data_file)
        print("\tCompleted!")

    if not os.path.isdir(args.tokenizer_dir):
        os.mkdir(args.tokenizer_dir)
        download_and_extract('stanford-corenlp', args.tokenizer_dir)

    os.environ["CLASSPATH"] = "{}stanford-corenlp-full-{}/stanford-corenlp-{}.jar".format(args.tokenizer_dir,
                                                                                          args.tokenizer_date,
                                                                                          args.tokenizer_ver)


def summarize(args, cp, data):

    # load path to model checkpoint
    if cp != '':
        checkpoint = cp
    else:
        checkpoint = args.checkpoint
    pt = torch.load(checkpoint, map_location=lambda storage, loc: storage)

    # configure pre-trained bert
    config = BertConfig.from_json_file(args.bert_path)

    # configure the device id
    device = "cpu" if args.visible_gpus == "-1" else "cuda"

    # load the model using checkpoint
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config=config)
    model.load_cp(pt)
    model.eval()

    num_sentence = args.num_sentence

    # build the trainer model
    bertsum = Trainer(args, device, model)

    # run inference on trainer model
    bertsum.infer(data, num_sentence)


if __name__ == '__main__':

    # load argument parser
    parser = argparse.ArgumentParser()

    # type of encoder
    parser.add_argument("-encoder", default='classifier', type=str, choices=['classifier', 'dnn'])
    # hyper-parameters for dnn
    parser.add_argument("-num_units", default=128, type=int)
    parser.add_argument("-num_layers", default=2, type=int)
    # path to model checkpoint
    parser.add_argument("-checkpoint", default='../models/demo_ckpt_classifier.pt')
    # path to bert configuration
    parser.add_argument("-bert_path", default='../models/bert_config_uncased_base.json')
    # path to save summary result
    parser.add_argument("-save_path", default='../results')
    # temp dir to cache bert
    parser.add_argument("-temp_dir", default='../temp')
    # available gpu devices
    parser.add_argument("-visible_gpus", default="-1", type=str)
    # number of summary sentences
    parser.add_argument("-num_sentence", default=3, type=int)
    # block trigram for inference?
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)
    # block scores lower than mean?
    parser.add_argument("-block_lower", type=str2bool, nargs='?', const=True, default=True)
    # configure stanford corenlp
    parser.add_argument("-tokenizer_dir", type=str, default="../stanford-corenlp/")
    parser.add_argument("-tokenizer_date", type=str, default="2018-10-05")
    parser.add_argument("-tokenizer_ver", type=str, default="3.9.2")

    args = parser.parse_args()
    cp = args.checkpoint
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    # configure stanford corenlp
    configure_tokenizer(args)

    # Receive input from user
    data = get_input()

    # Summarize the input article
    start = time.time()
    summarize(args, cp, data)

    # Report the time
    end = time.time()
    time = end - start
    print("\n[INFO] It took %.2f seconds to summarize" % time)
