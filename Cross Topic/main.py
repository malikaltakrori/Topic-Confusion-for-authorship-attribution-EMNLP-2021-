"""
Datasets:  1.1.0
Transformers: 3.1.0
TF:  2.3.1
"""
import re
import datasets

print('Datasets: ', datasets.__version__)

import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# dataset
from datasets import load_dataset, DatasetDict

import transformers

print('Transformers:', transformers.__version__)

# for zero-shot
from transformers import pipeline
from transformers import TFBertForSequenceClassification, TFDistilBertForSequenceClassification, \
	TFRobertaForSequenceClassification

# for finetuning
import tensorflow as tf

print('TF: ', tf.__version__)

from tensorflow import keras
from transformers import PreTrainedTokenizerFast, BertTokenizerFast, RobertaTokenizerFast, DistilBertTokenizerFast


def classify_zero_shot(dataset: DatasetDict, classify: str, model: str, batch_size: int) -> float:

	dataset_tests = dataset["test"]
	num_batches = dataset_tests.num_rows // batch_size

	classifier = pipeline("zero-shot-classification", model=model, device=0)  # to utilize GPU
	hypothesis_template = 'This text is written by {}.'  # the template used in this demo

	# classify =
	candidate_labels = dataset_tests.features[classify].names[:4]
	score, i = 0, 0
	while i < num_batches:
		# for i in range(num_batches):
		start_idx = i * batch_size
		end_idx = start_idx + batch_size

		# articles = dataset_tests['article'][:5]
		articles = dataset_tests['article'][start_idx:end_idx]
		# true_candidates = [candidate_labels[x] for x in dataset_tests['author'][:5]]
		true_candidates = [candidate_labels[x] for x in dataset_tests[classify][start_idx:end_idx]]

		# should be false to get a softmax dist
		result = classifier(articles, candidate_labels,
							hypothesis_template=hypothesis_template,
							multi_class=False)
		predictions = [x['labels'][0] for x in result]

		score += len([1 for (y, y_) in zip(true_candidates, predictions) if y == y_])

		i += 1

	# last batch _usually less than full
	articles = dataset_tests['article'][start_idx:end_idx]
	if len(articles) > 0:
		true_candidates = [candidate_labels[x] for x in dataset_tests[classify][start_idx:end_idx]]

		# should be false to get a softmax dist
		result = classifier(articles, candidate_labels,
							hypothesis_template=hypothesis_template,
							multi_class=False)
		predictions = [x['labels'][0] for x in result]

		score += len([1 for (y, y_) in zip(true_candidates, predictions) if y == y_])

	return score / dataset_tests.num_rows


def classify_few_shot(HFdataset: DatasetDict, classify: str, model: str, epochs: int = 3, batch_size: int = 4) -> None:
	case = re.findall(r'cross_topic_\d', HFdataset.cache_files['train'][0]['filename'])[0]


	if model.lower() == 'distilbert-base-uncased':
		tokenizer = DistilBertTokenizerFast.from_pretrained(model)
		columns_to_return = ['input_ids', 'attention_mask', classify]

	elif model.lower() == 'bert-base-uncased':
		tokenizer = BertTokenizerFast.from_pretrained(model)
		columns_to_return = ['input_ids', 'token_type_ids', 'attention_mask', classify]

	elif model.lower() == 'roberta-base':
		tokenizer = RobertaTokenizerFast.from_pretrained(model)
		columns_to_return = ['input_ids', 'attention_mask', classify]

	else:
		tokenizer = None

	if classify in ['author', 'topic']:
		content_label = 'article'  # 'article
	elif classify in ['label', 'news']:
		content_label = 'text'
		classify = 'label'

	train_dataset = HFdataset['train'].map(lambda e: tokenizer(e[content_label], truncation=True, padding='max_length'),
										   batched=True)

	# columns_to_return = ['input_ids', 'attention_mask', classify]
	train_dataset.set_format(type='tensorflow', columns=columns_to_return)

	#
	features = {x: train_dataset[x].to_tensor(default_value=0, shape=[None, tokenizer.model_max_length]) for x in
				columns_to_return[:-1]}
	tf_train_dataset = tf.data.Dataset.from_tensor_slices((features, train_dataset[classify])).shuffle(1000).batch(batch_size)

	num_labels = len(np.unique(train_dataset[classify]))

	valid_dataset = HFdataset['validation'].map(lambda e: tokenizer(e[content_label],
																	truncation=True, padding='max_length'),
												batched=True)
	valid_dataset.set_format(type='tensorflow', columns=columns_to_return)
	features = {x: valid_dataset[x].to_tensor(default_value=0, shape=[None, tokenizer.model_max_length]) for x in
				columns_to_return[:-1]}
	tf_valid_dataset = tf.data.Dataset.from_tensor_slices((features, valid_dataset[classify])).batch(batch_size)

	tests_dataset = HFdataset['test'].map(lambda e: tokenizer(e[content_label], truncation=True, padding='max_length'),
										  batched=True)
	tests_dataset.set_format(type='tensorflow', columns=columns_to_return)

	features = {x: tests_dataset[x].to_tensor(default_value=0, shape=[None, tokenizer.model_max_length]) for x in
				columns_to_return[:-1]}
	tf_tests_dataset = tf.data.Dataset.from_tensor_slices((features, tests_dataset[classify])).batch(batch_size)

	if model.lower() == 'distilbert-base-uncased':
		classifier = TFDistilBertForSequenceClassification.from_pretrained(model, num_labels=num_labels)
	elif model.lower() == 'bert-base-uncased':
		classifier = TFBertForSequenceClassification.from_pretrained(model, num_labels=num_labels)
	elif model.lower() == 'roberta-base':
		classifier = TFRobertaForSequenceClassification.from_pretrained(model, num_labels=num_labels)
	else:
		classifier = None

	# freeze embeddings
	if model.startswith('bert'):
		classifier.layers[0].trainable = False
		classifier.layers[1].trainable = False  # there's a dropout layer here
	elif model.startswith('roberta'):
		classifier.layers[0].trainable = False

	lr = 0.01

	classifier.compile(
		loss=classifier.compute_loss,
		optimizer=keras.optimizers.Adam(learning_rate=lr),
		metrics=["accuracy"],
	)

	classifier.summary()
	classifier.fit(tf_train_dataset, batch_size=batch_size, epochs=epochs, validation_data=tf_valid_dataset,
				   callbacks=[
					   tf.keras.callbacks.EarlyStopping(
						   # Stop training when `val_loss` is no longer improving
						   monitor="val_loss",
						   # "no longer improving" being defined as "no better than 1e-3 less"
						   min_delta=1e-3,
						   # "no longer improving" being further defined as "for at least 5 epochs"
						   patience=100,
						   verbose=1,
						   restore_best_weights=True,
					   ),
					   tf.keras.callbacks.ModelCheckpoint(
						   './weights/{}.{}.best_weights.hdf5'.format(model.split('-')[0], case[-1]),
						   monitor='val_loss', verbose=1, save_best_only=True,
						   save_weights_only=True, mode='min', save_freq='epoch'
					   )
				   ]
				   )

	print("testing")
	test_scores = classifier.evaluate(tf_tests_dataset, verbose=1)
	print("Test loss:", test_scores[0])
	print("Test accuracy:", test_scores[1])

	return test_scores[1]


def main():
	# region Define parser
	parser = argparse.ArgumentParser(
		description='Finetuning a pretrained LM and using it for text classification'
	)

	parser.add_argument('--case', choices=['cross_topic_1', 'cross_topic_2', 'cross_topic_3', 'cross_topic_4',
										   'cross_topic_5', 'cross_topic_6', 'cross_topic_7', 'cross_topic_8',
										   'cross_topic_9', 'cross_topic_10', 'cross_topic_11', 'cross_topic_12',
										   'cross_genre_1', 'cross_genre_2', 'cross_genre_3', 'cross_genre_4'
										   ],
						default='cross_topic_1')

	parser.add_argument('--classify', choices=['author', 'topic', 'label', 'news'],
						default='author')  # label for sentiment

	parser.add_argument('--model', choices=['bert', 'roberta', 'test'], default='Bert')

	parser.add_argument('--finetune', dest='finetune', action='store_true')
	parser.add_argument('--no-finetune', dest='finetune', action='store_false')
	parser.set_defaults(finetune=False)

	parser.add_argument('--epochs', type=int, default=3)

	parser.add_argument('--batch_size', type=int, default=8)

	args = parser.parse_args()
	# endregion


	# region Parse options and call trainers

	ds_path = './myDatasets/guardian_authorship_v2'

	if args.classify == 'author':
		HF_dataset = load_dataset(ds_path, name=args.case)
	elif args.classify == 'topic':
		HF_dataset = load_dataset(ds_path, name=args.case)
		HF_dataset['train'] = load_dataset(ds_path, name=args.case,
										   split='train[:60%]+validation[:60%]+test[:60%]')
		HF_dataset['validation'] = load_dataset(ds_path, name=args.case,
												split='train[60%:80%]+validation[60%:80%]+test[60%:80%]')
		HF_dataset['test'] = load_dataset(ds_path, name=args.case,
										  split='train[-20%:]+validation[-20%:]+test[-20%:]')
	elif args.classify == 'news':
		HF_dataset = load_dataset("ag_news")

	# HF_dataset = load_dataset('imdb')

	model = ''
	if str(args.model).lower() == 'test'.lower():
		model = 'distilbert-base-uncased'

	elif str(args.model).lower() == 'Bert'.lower():
		model = 'bert-base-uncased'  # Total params: 109,492,237

	elif str(args.model).lower() == 'Roberta'.lower():
		model = 'roberta-base'

	if not args.finetune:
		score = classify_zero_shot(HF_dataset, args.classify, model, args.batch_size)

	else:
		score = classify_few_shot(HF_dataset, args.classify, model, args.epochs, args.batch_size)


	# print('Model:{}, finetune:{}, case:{}, score:{}'.format(args.model, args.finetune, args.case, score))

	# endregion

	return 0


if __name__ == '__main__':
	main()
