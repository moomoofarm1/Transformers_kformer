import numpy as np
import pandas as pd
import ast
import torch
from os.path import isfile, join
from os import listdir
from seqeval.metrics import f1_score, precision_score, recall_score
import pysbd

from datasets import Dataset, DatasetDict
from transformers import (
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    AutoConfig,
    RobertaTokenizerFast,
    RobertaForTokenClassification
)


def read_tsv(file):
    columns = ["filename", "label", "off0", "off1", "span"]

    # Read txt sep by tabs
    df = pd.read_csv(file, sep="\t")

    for col in df.columns:
        if col not in columns:
            df.drop(col, axis=1, inplace=True)

    # Convert offsets to int
    df['off0'] = pd.to_numeric(df['off0'], errors="coerce")
    df['off1'] = pd.to_numeric(df['off1'], errors="coerce")

    # Sort by off0
    df = df.sort_values(by=['filename'])

    return df


def get_all_labels(path_dir):
    labels = set()
    files = [f for f in listdir(path_dir) if isfile(join(path_dir, f))]
    for file in files:
        if 'tsv' in file:
            for row in read_tsv(path_dir + file).iterrows():
                labels.add(row[1]['label'])

    final_labels = list(labels)

    # Add 'begin' and 'intern' label for every label
    for i in range(0, len(labels) * 2, 2):
        begin = 'B-' + final_labels[i]
        intern = 'I-' + final_labels[i]
        final_labels[i] = begin
        final_labels.insert(i + 1, intern)

    # Add 'O', [CLS] and [SEP] tokens
    final_labels.insert(0, 'O')
    final_labels += ['[CLS]', '[SEP]']

    return final_labels


def get_model(checkpoint, num_labels, label_map, label2id_, knowledge=None, from_tf=False):
    if knowledge is not None:
        config = AutoConfig.from_pretrained(
            checkpoint,
            num_labels=num_labels,
            id2label=label_map,
            label2id=label2id_,
            knowledge=knowledge
        )
    else:
        config = AutoConfig.from_pretrained(
            checkpoint,
            num_labels=num_labels,
            id2label=label_map,
            label2id=label2id_,
            knowledge=[-1, -1]
        )

    # model = AutoModelForTokenClassification.from_pretrained(
    #     checkpoint,
    #     config=config,
    #     from_tf=from_tf
    # )

    model = RobertaForTokenClassification.from_pretrained(
        checkpoint,
        config=config
    )

    try:
        dev = "cuda:1"
        device = torch.device(dev)
        model = model.to(device)
    except:
        print(f"No ha sido posible pasar el modelo a {dev}")

    tokenizer = RobertaTokenizerFast.from_pretrained(
        checkpoint,
        add_prefix_space=True
    )

    return config, model, tokenizer


def get_dataset(path, dir_paths, data_type, label2id_, add_know=False):
    columns = ["tokens", "labels", "knowledge"] if add_know else ["tokens", "labels"]
    datasets_dict = dict()
    dfs = dict()

    for d_type, directory in zip(data_type, dir_paths):
        csv_files = [f for f in listdir(path + directory) if isfile(join(path + directory, f))]
        df_col = pd.DataFrame(columns=columns)

        for csv_file in csv_files:
            df_readed = pd.read_csv(path + directory + csv_file)
            for i in range(len(columns)):
                df_readed[columns[i]] = df_readed[columns[i]].apply(ast.literal_eval)
            df_col = pd.concat([df_col, df_readed])

        df_col = df_col.reset_index(drop=True)

        # Cambiar esto por algo más eficiente o por lo menos más bonito
        for i in range(len(df_col)):
            list_label = [label2id_[df_col.iloc[i]['labels'][j]] for j in range(len(df_col.iloc[i]['labels']))]
            df_col.at[i, 'labels'] = list_label

        datasets_dict[d_type] = Dataset.from_pandas(df_col)

    if "train" in data_type and "dev" not in data_type and "test" not in data_type:
        datasets = DatasetDict({data_type[0]: datasets_dict[data_type[0]]})
    elif "train" in data_type and "dev" in data_type and "test" not in data_type:
        datasets = DatasetDict({data_type[0]: datasets_dict[data_type[0]],
                                data_type[1]: datasets_dict[data_type[1]]})
    elif "train" in data_type and "dev" in data_type and "test" in data_type:
        datasets = DatasetDict({data_type[0]: datasets_dict[data_type[0]],
                                data_type[1]: datasets_dict[data_type[1]],
                                data_type[2]: datasets_dict[data_type[2]]})
    elif "sample" in data_type:
        datasets = DatasetDict({data_type[0]: datasets_dict[data_type[0]]})
    else:
        return False

    return datasets


def tokenize_and_align_labels(examples):
    # True: expand the label to the division of a token
    # False: set -100 label to the divison of a token
    label_all_tokens = True

    tokenized_inputs = _tokenizer(examples["tokens"],
                                  truncation=True,
                                  is_split_into_words=True)

    tokenized_knowledge = []
    for row in examples["knowledge"]:
        tokenized_row = {'input_ids': [], 'attention_mask': []}
        for know in row:
            tokenized_know = (_tokenizer(know,
                                         truncation=True))
            tokenized_row['input_ids'].append(tokenized_know['input_ids'])
            tokenized_row['attention_mask'].append(tokenized_know['attention_mask'])
        tokenized_knowledge.append(tokenized_row)

    labels = []
    for i, label in enumerate(examples[f"labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                # label_ids.append(label[word_idx] if label_all_tokens else -100)
                if label_all_tokens:
                    if label[word_idx] == 0:
                        label_ids.append(label[word_idx])
                    else:
                        # If label is odd it means is a B- (begin) label so add one to take I- (intern) label
                        # If label is even it means is a I- (intern) label
                        if label[word_idx] % 2 - 1 == 0:
                            label_ids.append(label[word_idx] + 1)
                        else:
                            label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    l = []
    for e in tokenized_knowledge:
        l.append(e['input_ids'])
    tokenized_inputs["knowledge"] = l
    return tokenized_inputs


def align_predictions(predictions: np.ndarray, label_ids: np.ndarray, label_map):
    preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != torch.nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list, out_label_list


def compute_metrics(p: EvalPrediction, label_map):
    preds_list, out_label_list = align_predictions(p.predictions, p.label_ids, label_map)

    return {
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }


def get_trainer(path, model, tokenizer, tokenized_datasets, data_collator):
    training_args = TrainingArguments(path + "test-trainer",
                                      num_train_epochs=2.0,
                                      )

    trainer = Trainer(model,
                      training_args,
                      train_dataset=tokenized_datasets["train"],
                      eval_dataset=tokenized_datasets["dev"],
                      data_collator=data_collator,
                      tokenizer=tokenizer,
                      compute_metrics=compute_metrics,
                      )

    return training_args, trainer


def tokenize_test(entrada, tokenizer):
    tokenized_knowledge = []
    tokenized_row = {'input_ids': [], 'attention_mask': []}
    for know in entrada['knowledge']:
        tokenized_know = (tokenizer(know, truncation=True))
        tokenized_row['input_ids'].append(tokenized_know['input_ids'])
        tokenized_row['attention_mask'].append(tokenized_know['attention_mask'])
    tokenized_knowledge.append(tokenized_row)

    encoding = tokenizer(entrada['tokens'], return_tensors="pt", is_split_into_words=True, truncation=True)

    l = []
    for e in tokenized_knowledge:
        l.append(e['input_ids'])
    # encoding["knowledge"] = torch.tensor(l)

    return encoding


def pred(model, encoding):
    # entrada = {'tokens': ['El', '1', 'de', 'enero', 'de', '2020,', 'ingresó', 'en', 'el', 'Union', 'Hospital', '(facultad', 'de',
    #                       'medicina', 'Tongji,', 'Wuhan,', 'provincia', 'de', 'Hubei)', 'un', 'hombre', 'de', '42', 'años', 'con',
    #                       'hipertermia', '(39,6', '°C),', 'tos', 'y', 'que', 'refería', 'fatiga', 'de', 'una', 'semana', 'de',
    #                       'evolución', '.'],
    #            'knowledge': ['This is a test 0', 'This is a test 1', 'This is a test 2', 'This is a test 3', 'This is a test 4',
    #                          'This is a test 5', 'This is a test 6', 'This is a test 7', 'This is a test 8', 'This is a test 9',
    #                          'This is a test 10', 'This is a test 11', 'This is a test 12', 'This is a test 13',
    #                          'This is a test 14', 'This is a test 15', 'This is a test 16', 'This is a test 17',
    #                          'This is a test 18', 'This is a test 19']}

    outputs = model(**encoding)
    logits = outputs.logits
    # print(logits.shape)
    predicted_label_classes = logits.argmax(-1)
    # print(predicted_label_classes)
    predicted_labels = [model.config.id2label[id] for id in predicted_label_classes.squeeze().tolist()]
    print(predicted_labels, end="\n"*2)

    return predicted_labels


def model_test():
    from transformers import pipeline

    path = "C:/Users/carlo/Documents/Máster/Trabajo-Fin-De-Master/LivingNER/"
    path_data = path + "data/"
    path_test = path_data + "training_valid_test_background_multilingual/test_background/pruebas/"
    path_tsv = path_test + "prueba_test.tsv"

    # Estas labels están al revés
    labels = ['O', 'B-SPECIES', 'I-SPECIES', 'B-HUMAN', 'I-HUMAN', '[CLS]', '[SEP]']
    label_map = {i: label for i, label in enumerate(labels)}
    label2id_ = {label: i for i, label in enumerate(labels)}
    num_labels = len(labels)

    files = [f for f in listdir(path_test) if isfile(join(path_test, f))]

    checkpoint = "C:/Users/carlo/Documents/Máster/Trabajo-Fin-De-Master/LivingNER/models/checkpoint-9500"
    # checkpoint = "PlanTL-GOB-ES/roberta-base-bne"

    _, model, tokenizer = get_model(checkpoint, num_labels, label_map, label2id_)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    df = pd.DataFrame(columns=["filename", "mark", "label", "off0", "off1", "span"])

    for file in files:
        predictions = []

        with open(path_test + file, 'r', encoding="utf-8-sig") as f:
            text = f.read()

        seg = pysbd.Segmenter(language="es", clean=False)
        sentences = seg.segment(text)

        for sentence in sentences:
            # Aquí se pillaría el knowledge de donde sea, ahora se hace una prueba solo
            knowledge = []
            for i in range(20):
                knowledge.append(f"This is a test {i}")

            tokenized_knowledge = []
            for know in knowledge:
                tokenized_know = (tokenizer(know, truncation=True))
                tokenized_knowledge.append(tokenized_know['input_ids'])
            tokenized_knowledge = torch.LongTensor([tokenized_knowledge])
            # ********************************************************************
            prediction = nlp(sentence, ignore_labels=[], grouped_entities=True)
            # prediction = nlp(sentence, knowledge=tokenized_knowledge, ignore_labels=[], grouped_entities=True)
            predictions.append({'prediction': prediction, 'sentence': sentence})

        df = pd.concat([df, add_predictions_to_df(df, file, text, predictions)], ignore_index=True)

    return None


def add_predictions_to_df(df, file, text, predictions):
    """
    [{'end': 5,
      'entity_group': 'SPECIES',
      'score': 0.9974855,
      'start': 0,
      'word': ' Mujer'},
     {'end': 35,
      'entity_group': 'O',
      'score': 0.9997915,
      'start': 6,
      'word': ' de 59 años cuyos antecedentes'},
     {'end': 46,
      'entity_group': 'SPECIES',
      'score': 0.99742824,
      'start': 36,
      'word': ' personales'},
     """
    pred_dict = {"filename": [], "mark": [], "label": [], "off0": [], "off1": [], "span": []}

    #
    # PROBLEMA
    # PROBLEMA
    #
    # SARS-COV-2 O LO QUE SEA LO ESTÁ SEPARANDO PORQUE ENTIENDE QUE TODO ESO ES B-SPECIE
    # ESO HABRÍA QUE CAMBIARLO EN LO DE EXPANDIR
    #
    # DONE - OTRA COSA: EN EL PROCESADO DE DATOS HACER QUE SEA UTF-8-SIG PARA ELIMINAR LA MOVIDA DE PRINCIPIO DE FRASE
    #

    mark = 1
    for element in predictions:
        prediction = element['prediction']
        for entity in prediction:
            if entity['entity_group'] != 'O':
                sentence = element['sentence']
                pred_dict['filename'].append(file)
                pred_dict['mark'].append(f"T{mark}")
                pred_dict['label'].append(entity['entity_group'])
                pred_dict['span'].append(entity['word'])

                init_s = text.find(sentence)
                pred_dict['off0'].append(init_s + entity['start'])
                pred_dict['off1'].append(init_s + entity['end'])

                mark += 1

    return pd.DataFrame.from_dict(pred_dict)


def prueba_pipeline():
    from transformers import pipeline

    labels = ['O', 'B-HUMAN', 'I-HUMAN', 'B-SPECIES', 'I-SPECIES', '[CLS]', '[SEP]']
    label_map = {i: label for i, label in enumerate(labels)}
    label2id_ = {label: i for i, label in enumerate(labels)}
    num_labels = len(labels)

    # checkpoint = "PlanTL-GOB-ES/roberta-base-bne"
    checkpoint = "C:/Users/carlo/Documents/Máster/Trabajo-Fin-De-Master/LivingNER/models/checkpoint-9500"

    _, model, tokenizer = get_model(checkpoint, num_labels, label_map, label2id_)

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    text = "Mujer de 59 años cuyos antecedentes personales incluyen hipertensión arterial, artropatía degenerativa cervical, lumbociática crónica, tuberculosis ganglionar diagnosticada por cuadro de eritema nudoso y migraña."
    knowledge = []
    for i in range(20):
        knowledge.append(f"This is a test {i}")

    tokenized_knowledge = []
    for know in knowledge:
        tokenized_know = (tokenizer(know, truncation=True))
        tokenized_knowledge.append(tokenized_know['input_ids'])
    tokenized_knowledge = torch.LongTensor([tokenized_knowledge])

    # res = nlp(text, grouped_entities=True)
    res = nlp(text, knowledge=tokenized_knowledge, ignore_labels=[], grouped_entities=True)

    for i in res:
        print(i)

    return res


if __name__ == "__main__":
    # model_test()
    #
    # exit(0)

    _path = "C:/Users/carlo/Documents/Máster/Trabajo-Fin-De-Master/LivingNER/"
    _path_code = _path + "Code/"
    _path_data = _path + "data/"
    _path_models = _path + "Models/"
    _path_knowledge = "C:/Users/carlo/Documents/Máster/Trabajo-Fin-De-Master/Data/knowledge_prueba"

    num_threads = 16

    _labels = get_all_labels(_path_data + "train-set/subtask1-NER/")
    print(_labels)
    _label_map = {i: label for i, label in enumerate(_labels)}
    _label2id_ = {label: i for i, label in enumerate(_labels)}
    _num_labels = len(_labels)

    _train_path = "train-set-pruebas/processed/"
    _dev_path = "dev-set-pruebas/processed/"
    _test_path = "test-set-pruebas/processed/"

    _dir_paths = [_train_path, _dev_path]
    _data_type_ = ["train", "dev"]

    print("Cargando datos...")

    _raw_datasets = get_dataset(_path_data, _dir_paths, _data_type_, _label2id_, add_know=True)

    print("DONE\n")

    # checkpoint = "/content/drive/MyDrive/Golvin/LivingNER/Models/pt_save_pretrained_1"
    _checkpoint = "PlanTL-GOB-ES/roberta-base-bne"

    print("Obteniendo modelo y tokenizer...")

    _, _model, _tokenizer = get_model(_checkpoint, _num_labels, _label_map, _label2id_)

    print("DONE\n")

    _tokenized_datasets = _raw_datasets.map(tokenize_and_align_labels, batched=True)

    _data_collator = DataCollatorForTokenClassification(tokenizer=_tokenizer, padding=True)

    _, _trainer = get_trainer(_path, _model, _tokenizer, _tokenized_datasets, _data_collator)

    print("Entrenando...")

    _trainer.train()

    print("DONE\n")
