import logging
import os

import torch
import wordninja
from PIL import Image
from torchvision import transforms

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, text, img_id, label=None):
        """Constructs an InputExample."""
        self.text = text
        self.img_id = img_id
        self.label = label


class MMInputFeatures(object):
    def __init__(self, input_ids,
                 input_mask,
                 added_input_mask,
                 img_feat,
                 objtag_input_ids,
                 objtag_input_mask,
                 label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.added_input_mask = added_input_mask
        self.img_feat = img_feat
        self.objtag_input_ids = objtag_input_ids
        self.objtag_input_mask = objtag_input_mask
        self.label_id = label_id


class Processer():
    def __init__(self, data_dir, image_path, model_select, max_seq_length, max_objtag_length):
        self.model_select = model_select
        self.max_seq_length = max_seq_length
        self.max_objtag_length = max_objtag_length
        self.image_path = image_path
        self.data_dir = data_dir

    def get_train_examples(self):
        return self._create_examples(os.path.join(self.data_dir, "train.txt"))

    def get_eval_examples(self):
        return self._create_examples(os.path.join(self.data_dir, "valid.txt"))

    def get_test_examples(self):
        return self._create_examples(os.path.join(self.data_dir, "test.txt"))

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, data_file):
        """Creates examples for the training and dev sets."""
        examples = []
        with open(data_file) as f:
            for line in f.readlines():
                lineLS = eval(line)
                try:
                    tmpLS = lineLS[1].split()
                except:
                    print('tmpLS:continue')
                    continue
                try:
                    img_id = lineLS[0]
                    text = lineLS[1]
                    label = int(lineLS[-1])
                    examples.append(InputExample(text=text, img_id=img_id, label=label))
                except:
                    print('other:continue')
                    continue
        return examples

    def image_process(self, image_path, transform):
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        return image

    def get_objtag_text(self):
        objtag = {}
        with open(self.image_path) as f:
            for line in f.readlines():
                sp = line.strip().split()
                if sp[0] not in objtag.keys():
                    objtag[sp[0]] = " ".join(sp)
        return objtag

    def convert_mm_examples_to_features(self, examples, label_list, tokenizer):
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        for (ex_index, example) in enumerate(examples):

            sent = example.text.split()

            image_text_dic = self.get_objtag_text()

            if example.img_id in image_text_dic:
                objtags = list(image_text_dic[example.img_id])
            else:
                objtags = ["None"]


            tokens = tokenizer.tokenize(sent)
            objtags = tokenizer.tokenize(objtags)

            if len(tokens) > self.max_seq_length - 2:
                tokens = tokens[:(self.max_seq_length - 2)]
            if len(objtags) > self.max_objtag_length - 2:
                objtags = objtags[:(self.max_objtag_length - 2)]

            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            added_input_mask = [1] * (len(input_ids) + 49)
            padding = [0] * (self.max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            added_input_mask += padding

            objtags = ["[CLS]"] + objtags + ["[SEP]"]
            objtag_input_ids = tokenizer.convert_tokens_to_ids(objtags)
            objtag_input_mask = [1] * len(objtag_input_ids)
            objtag_padding = [0] * (self.max_objtag_length - len(objtag_input_ids))
            objtag_input_ids += objtag_padding
            objtag_input_mask += objtag_padding
            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length

            assert len(objtag_input_ids) == self.max_objtag_length
            assert len(objtag_input_mask) == self.max_objtag_length
            label_id = label_map[example.label]

            # process images
            image_name = example.img_id
            image_path = os.path.join(self.image_path, image_name + ".jpg")
            image = self.image_process(image_path, transform)  # 3*224*224

            features.append(MMInputFeatures(input_ids=input_ids,
                                            input_mask=input_mask,
                                            added_input_mask=added_input_mask,
                                            img_feat=image,
                                            objtag_input_ids=objtag_input_ids,
                                            objtag_input_mask=objtag_input_mask,
                                            label_id=label_id))
            if ex_index % 1000 == 0:
                logger.info("processed image num: " + str(ex_index) + " **********")
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in features], dtype=torch.long)
        all_img_feats = torch.stack([f.img_feat for f in features])
        all_objtag_input_ids = torch.tensor([f.objtag_input_ids for f in features], dtype=torch.long)
        all_objtag_input_mask = torch.tensor([f.objtag_input_mask for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_added_input_mask, all_img_feats, all_objtag_input_ids, all_objtag_input_mask, all_label_ids
