import re
import json
import numpy as np
import torch.utils.data as data
from utils.get_tokenizer import get_tokenizer
import pickle


class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.base_dir = args.base_dir
        self.tokenizer, _ =  get_tokenizer(args.llm_model, args.num_tokens)
        features = pickle.load(open(self.args.feature_path, 'rb'))
        features = features['train'] + features['test']
        self.features = {i['id']:i['feature'] for i in features}

    def tokenize(self, text):
        out = self.tokenizer(
            text,
            return_tensors="pt",
            padding='longest')
        input_ids = out.input_ids[0]
        len_caption = out.attention_mask[0].sum()
        return input_ids, len_caption


    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                               .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        report = ' '.join(report.split()[:self.args.max_length])
        return report
    

    def parse(self, features):
        report = features.get("report", "")
        report = self.clean_report_mimic_cxr(report)

        # VQGAN encoder embedding
        image_embed = self.features[features['id']]

        report_with_vid = report
        for i in range(self.args.num_tokens):
            report_with_vid += f'[IMG{i}]'

        input_ids_with_vid, len_report_with_vid = self.tokenize(report_with_vid)

        to_return = {
            "id": features['id'],
            "image_path": features['image_path'][0],
            "image_emb": image_embed,
            "report": report_with_vid,
            "input_ids": input_ids_with_vid,
            "len_report": len_report_with_vid
        }
        return to_return


    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        # pdb.set_trace()
        self.train = split == "train"
        meta = json.load(open(args.dataset, 'r'))
        if split == "train":
            self.df = meta['train']
        else:
            self.df = meta['test']
        self.parser = FieldParser(args)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        try:
            return self.parser.transform_with_parse(self.df[index])
        except Exception as e:
            print(f'Error reading for {self.df[index]["id"]}: {e}')
            # Pick a new example at random.
            idx = np.random.randint(0, len(self.df)-1)


def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset



if __name__ == '__main__':
    from configs.config import parser
    args = parser.parse_args()
    loader = ParseDataset(args)

    data = loader.__getitem__(12)


