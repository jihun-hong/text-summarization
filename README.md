# Extractive Summarization
This is the code that implements inference model for BERT + Classifier. The algorithm fine-tunes Google's language model BERT (Bidirectional Encoder Representations from Transformers) for extractive text summarization, and enables the user to do end-to-end inference with the saved model. The pytorch model can be optimized for inference.

The original code for BERTSUM by Yang Liu could be found [here](https://github.com/nlpyang/BertSum)

**Python version**: the code is written in Python 3.6

**Package Requirements**: pytorch, pytorch_pretrained_bert

## Stanford CoreNLP
The model uses Stanford CoreNLP to tokenize the raw input text during the preprocessing phase.
Running the code will automatically download the Stanford CoreNLP module to the "stanford-corenlp" directory.
If you already have Stanford CoreNLP downloaded, put the files inside the "stanford-corenlp" directory.

* Note that the code will default to downloading the **3.9.2 version** of Stanford CoreNLP, which was released on 2018-10-05. If you want to change the version, you can do so by altering the command line arguments, as noted below.

* Also, the current repository doesn't have the stanford-corenlp directory. Running the code will create the repository, and download Stanford-CoreNLP files inside the folder. If you already have CoreNLP downloaded, please create a directory named "stanford-corenlp" and put the entire folder insider the directory.

## Training and Evaluation
This model is only for BERTSUM at the inference level. For model training and model evaluation, please refer to the original BERTSUM code, which could be found [here](https://github.com/nlpyang/BertSum). You should follow the steps from Yang Liu's repository to train and save the model. Once the model is finished training and saved, you might want to shrink the size of the model in order to speed up the inference. The following code shrinks the model size, and makes inference 25% faster.

#### Optimizing the model
In order to optimize the model, please run the following code:

```
python optimize.py -path PATH -new_name NAME
```

⚠️ PATH should be replaced with the actual path to the checkpoint model you saved. NAME should be replaced with the new name you intend to give to the new model. The file will be saved under the same folder as your original model, with the new given name.

🚀 Optimizing the model will boost the inference speed by 40% and shrink the model size by 70%. If you train your own model using Yang Liu's BertSum code, please use this optimization feature before you run inference on the model.

## Choosing the Encoder
In order to run the model, you must download the checkpoint for BERT + Classifier in the "model" repository. There are two option for the summarization layer. The first is the Simple Classifier, which is used in the original BertSum paper by Yang Liu. The second is the Deep Classifier, which uses deep feed-forward network as the summarization layer.

Accordingly, there are separate checkpoint files available for each summarization layer. The checkpoint for Simple Classifier is provided by Yang Liu. The checkpoint for Deep Classifier was trained by Jihun Hong, using the same dataset and method as the original BertSum paper. You could make inference with both encoders using the command line arguments.

* BERT + Simple Classifier : [link](https://drive.google.com/file/d/1VN4tuWeRcFqEv4J1Xb7BIIr9Ym4SGOVz/view?usp=sharing)

* BERT + Deep Classifier : [link](https://drive.google.com/file/d/1v_LreKIRiEAieRI4cnD_LnFHX6cSfg4X/view?usp=sharing)

The saved checkpoints are optimized for inference. Compared to the fully loaded checkpoints produced during training phase, these saved checkpoints have 70% less file size, as well as show 20% faster inference speed.

## Running Inference
Type in the below code to run inference.

```
python inference.py 
```

This will prompt you to input the raw text that you wish to summarize. You could feed in the text that you wish to summarize, and press enter. The model will return to you the summary as text, with the number of sentences specified by the user. The summary is also saved as a .text file in the "results" directory.

The default option for extractive summarization is as follows:

```
python inference.py -num_sentence 3 -block_trigram true -visible_gpus -1 -checkpoint ../models/demo_ckpt_classifier.pt -bert_path '../models/bert_config_uncased_base.json' -tokenized_dir '../stanford-corenlp' -tokenizer_date '2018-10-05' -tokenized_ver '3.9.2'
```

* If you want to run inference on GPU, enter the following: ```-visible_gpus 0```

* If you want to turn off trigram blocking, enter the following: ```-block_trigram false```

* If you want five sentences for the summary, enter the following: ```-num_sentence 5```

If you want to use the dnn encoder for summarization, use the following:

```
python inference.py -encoder dnn -checkpoint ../models/demo_ckpt_dnn.pt
```

UPDATE (2019/07/17):

We added a feature "block_lower" to the model. Setting ```-block_lower true``` will not include sentences that have scores less than the arithmetic mean of the entire sentence scores array. For example, let's say that the sentence scores for a text with 5 sentences is [0.4, 0.1, 0.9, 0.3, 0.8]. The aritmetic mean of the array is 0.5, so this means that sentences with scores that are less than 0.5 will not be included in the summary, no matter how large ```num_sentence``` is.

## Example Log
```
(base) C:\Users\42maru\PycharmProjects\demo_bertsum_en\src>python inference.py
[INFO] Starting BERTSUM ...
[INFO] Enter the text:

When I first brought my cat home from the Humane Society she was a mangy, pitiful animal. She was so thin that you could count her vertebrae just by looking at her. Apparently she was declawed by her previous owners, then abandoned or lost. Since she couldn't hunt, she nearly starved. Not only that, but she had an abscess on one hip. The vets at the Humane Society had drained it, but it was still scabby and without fur. She had a terrible cold, too. She was sneezing and sniffling and her meow was just a hoarse squeak. And she'd lost half her tail somewhere. Instead of tapering gracefully, it had a bony knob at the end.

[INFO] Processing text using Stanford CoreNLP ...

Adding annotator tokenize
Adding annotator ssplit

Processing file C:\Users\42maru\PycharmProjects\demo_bertsum_en\src\..\temp\tokenized.story ... writing to C:\Users\42maru\PycharmProjects\demo_bertsum_en\temp\tokenized.story.json
Annotating file C:\Users\42maru\PycharmProjects\demo_bertsum_en\src\..\temp\tokenized.story ... done [0.1 sec].

Annotation pipeline timing information:
TokenizerAnnotator: 0.1 sec.
WordsToSentencesAnnotator: 0.0 sec.
TOTAL: 0.1 sec. for 136 tokens at 2472.7 tokens/sec.
Pipeline setup: 0.0 sec.
Total time for StanfordCoreNLP pipeline: 0.2 sec.

[INFO] Successfully tokenized text using Stanford CoreNLP

[INFO] This is the summary:

The vets at the Humane Society had drained it, but it was still scabby and without fur. Not only that, but she had an abscess on one hip. Apparently she was declawed by her previous owners, then abandoned or lost.

[INFO] The summary produced 3 sentences

[INFO] It took 1.52 seconds to summarize
```
