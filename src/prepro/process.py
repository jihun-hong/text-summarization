import os
import json
import subprocess

from prepro.utils import clean
from pytorch_pretrained_bert import BertTokenizer

MAX_TOKENS = 510

def processor(text):

    print("\n[INFO] Processing text using Stanford CoreNLP ...\n")
    tokenized_dir = tokenize(text)
    source = format_to_line(tokenized_dir)
    data = format_to_bert(source)

    return data


def tokenize(text):
    # Write the raw file
    raw_file = write_to(text)

    # Tokenize raw text using CoreNLP
    tokenized_dir = "../temp/"
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize, ssplit',
               '-ssplit.newlineIsSentenceBreak', 'always', '-file', raw_file,
               '-outputFormat', 'json', '-outputDirectory', tokenized_dir]
    subprocess.call(command)

    os.remove(raw_file)
    tokenized_dir = tokenized_dir + raw_file.split("/")[-1] + ".json"

    print("\n[INFO] Successfully tokenized text using Stanford CoreNLP")
    return tokenized_dir


def write_to(text):
    file_name = "../temp/tokenized.story"
    with open(file_name, "w") as f:
        f.write(text)

    return file_name


def format_to_line(tokenized_dir):

    src = []

    for sent in json.load(open(tokenized_dir))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        src.append(tokens)
    source = [clean(' '.join(sent)).split() for sent in src]

    os.remove(tokenized_dir)

    return source


def format_to_bert(source):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        sep_vid = tokenizer.vocab['[SEP]']
        cls_vid = tokenizer.vocab['[CLS]']

        src = source
        src_text = [' '.join(sent) for sent in src]
        text = ' [SEP] [CLS] '.join(src_text)

        src_subtokens = tokenizer.tokenize(text)
        if len(src_subtokens) > MAX_TOKENS:
            print("\n[INFO] The text is too long to summarize ...\n")
            print("[ERROR] The text has %d tokens, while the maximum is 510 tokens ...\n" % len(src_subtokens))
            return
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']

        src_ids = tokenizer.convert_tokens_to_ids(src_subtokens)

        _sep = [-1] + [i for i, t in enumerate(src_ids) if t == sep_vid]
        sep = [_sep[i] - _sep[i - 1] for i in range(1, len(_sep))]

        segs = []
        for i, s in enumerate(sep):
            if i % 2:
                segs += s * [1]
            else:
                segs += s * [0]

        clss = [i for i, t in enumerate(src_ids) if t == cls_vid]

        data = {'src': src_ids, 'segs': segs, 'clss': clss, 'src_str': src_text}
        return data
