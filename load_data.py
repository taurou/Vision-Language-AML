from PIL import Image
import torch
import json
import random
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as T

CATEGORIES = {
    'dog': 0,
    'elephant': 1,
    'giraffe': 2,
    'guitar': 3,
    'horse': 4,
    'house': 5,
    'person': 6,
}

description_titles = ["Level of details:" , "Edges:" , "Color saturation:", "Color shades:","Background:" ,"Single instance:" , "Text:" , "Texture:",  "Perspective:" ]


def tokenize_description(item, description_titles):
    concat_descr = ""
    for idx, descr in enumerate(item):
       concat_descr =  " ".join([concat_descr, description_titles[idx], descr]).replace("-", "").replace(":", "").replace(",", "") #We remove non-alphabetic chars because clip.tokenize counts them as a single word and the number of words grows quickly.
    return ' '.join(concat_descr.split(' ')[:75]) #Limit the number of description words to 77 because of clip.tokenize limits. #TODO TOBETESTED at the moment limited to 75 because some documentation on the internet says that the beginning and the end of the string count as individual words, thus 75+2 = 77

def create_dict(json_descr):
    dict = {}
    for item in json_descr:
        description = tokenize_description(item["descriptions"], description_titles).strip()
        dict[item["image_name"]] = description
    return dict

class PACSDatasetBaseline(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        img_path, y = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y

def read_lines(data_path, domain_name):
    examples = {}
    with open(f'{data_path}/{domain_name}.txt') as f:
        lines = f.readlines()

    for line in lines: 
        line = line.strip().split()[0].split('/')
        category_name = line[3]
        category_idx = CATEGORIES[category_name]
        image_name = line[4]
        image_path = f'{data_path}/kfold/{domain_name}/{category_name}/{image_name}'
        if category_idx not in examples.keys():
            examples[category_idx] = [image_path]
        else:
            examples[category_idx].append(image_path)
    return examples

def build_splits_baseline(opt):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']

    source_examples = read_lines(opt['data_path'], source_domain)
    target_examples = read_lines(opt['data_path'], target_domain)

    # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}

    # Build splits - we train only on the source domain (Art Painting)
    val_split_length = source_total_examples * 0.2 # 20% of the training split used for validation

    train_examples = []
    val_examples = []
    test_examples = []

    for category_idx, examples_list in source_examples.items():
        split_idx = round(source_category_ratios[category_idx] * val_split_length)
        for i, example in enumerate(examples_list):
            if i > split_idx:
                train_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
            else:
                val_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    
    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            test_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    
    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    train_loader = DataLoader(PACSDatasetBaseline(train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetBaseline(val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetBaseline(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader

def build_splits_domain_disentangle(opt):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']

    source_examples = read_lines(opt['data_path'], source_domain)
    target_examples = read_lines(opt['data_path'], target_domain)

    # Compute ratios of SOURCE examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}
    source_val_split_length = source_total_examples * 0.2 # 20% of the training split used for validation and test

    # Compute ratios of TARGET examples for each category
    target_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in target_examples.items()}
    target_total_examples = sum(target_category_ratios.values())
    target_category_ratios = {category_idx: c / target_total_examples for category_idx, c in target_category_ratios.items()}
    target_val_split_length = target_total_examples * 0.2 # 20% of the training split used for validation and test

    source_train_examples = []
    target_train_examples = []
    source_val_examples = []
    target_test_examples = []

    # Build splits - source domain (Art Painting)
    for category_idx, examples_list in source_examples.items():
        split_idx = round(source_category_ratios[category_idx] * source_val_split_length)
        for i, example in enumerate(examples_list):
            if i > split_idx:
                source_train_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
            else:
                source_val_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]

    # Build splits - target domain
    for category_idx, examples_list in target_examples.items():
        split_idx = round(target_category_ratios[category_idx] * target_val_split_length)
        for i, example in enumerate(examples_list):
            if i > split_idx:
                target_train_examples.append([example, category_idx]) # each pair is [path_to_img, class_label] the category label is inserted here but is not used in (unsupervised) training
            else:
                target_test_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]

    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    source_train_loader = DataLoader(PACSDatasetBaseline(source_train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    target_train_loader = DataLoader(PACSDatasetBaseline(target_train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    source_val_loader = DataLoader(PACSDatasetBaseline(source_val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetBaseline(target_test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return source_train_loader, target_train_loader, source_val_loader, test_loader


class PACSDatasetCLIP(Dataset):
    def __init__(self, examples, transform, descr_dict):
        self.examples = examples
        self.transform = transform
        self.descr_dict = descr_dict

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        img_path, y = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        key_dict = img_path.replace("data/PACS/kfold/", "")
        if key_dict in self.descr_dict:
            return x, y, self.descr_dict[key_dict]
        else:
            return x, y
    


class ConditionalBatchSampler(Sampler):
    def __init__(self, examples, descr_dict, batch_size, shuffle=True):
        self.examples = examples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.descr_dict = descr_dict

        self.indices = list(range(len(self.examples)))
        self.condition_indices = [idx for idx, item in enumerate(self.examples) if item[0].replace("data/PACS/kfold/", "") in descr_dict]
        self.non_condition_indices = [idx for idx, item in enumerate(self.examples) if item[0].replace("data/PACS/kfold/", "") not in descr_dict]
        self.num_condition_batches = len(self.condition_indices) // batch_size
        self.num_non_condition_batches = len(self.non_condition_indices) // batch_size

        if self.shuffle:
            random.seed(0)
            random.shuffle(self.non_condition_indices)
            random.shuffle(self.condition_indices)

    def __iter__(self):
        for i in range(min(self.num_condition_batches, self.num_non_condition_batches)):
            condition_batch_indices = self.condition_indices[i * self.batch_size:(i + 1) * self.batch_size]
            yield condition_batch_indices

            non_condition_batch_indices = self.non_condition_indices[i * self.batch_size:(i + 1) * self.batch_size]
            yield non_condition_batch_indices

            #batch_indices = condition_batch_indices + non_condition_batch_indices
            #yield [self.indices[idx] for idx in batch_indices]

    def __len__(self):
        return max(self.num_condition_batches, self.num_non_condition_batches)



def build_splits_clip_disentangle(opt):

    load_descriptions = json.load(open("./data/descriptions/all_labels_plus_assigned.json", "r"))
    descriptions = create_dict(load_descriptions)

    source_domain = 'art_painting'
    target_domain = opt['target_domain']

    source_examples = read_lines(opt['data_path'], source_domain)
    target_examples = read_lines(opt['data_path'], target_domain)

    # Compute ratios of SOURCE examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}
    source_val_split_length = source_total_examples * 0.2 # 20% of the training split used for validation and test

    # Compute ratios of TARGET examples for each category
    target_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in target_examples.items()}
    target_total_examples = sum(target_category_ratios.values())
    target_category_ratios = {category_idx: c / target_total_examples for category_idx, c in target_category_ratios.items()}
    target_val_split_length = target_total_examples * 0.2 # 20% of the training split used for validation and test

    source_train_examples = []
    target_train_examples = []
    source_val_examples = []
    target_test_examples = []

    # Build splits - source domain (Art Painting)
    for category_idx, examples_list in source_examples.items():
        split_idx = round(source_category_ratios[category_idx] * source_val_split_length)
        for i, example in enumerate(examples_list):
            if i > split_idx:
                source_train_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
            else:
                source_val_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]

    # Build splits - target domain
    for category_idx, examples_list in target_examples.items():
        split_idx = round(target_category_ratios[category_idx] * target_val_split_length)
        for i, example in enumerate(examples_list):
            if i > split_idx:
                target_train_examples.append([example, category_idx]) # each pair is [path_to_img, class_label] the category label is inserted here but is not used in (unsupervised) training
            else:
                target_test_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]

    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    source_train_loader = DataLoader(PACSDatasetCLIP(source_train_examples, train_transform, descriptions), batch_sampler=ConditionalBatchSampler(source_train_examples, descriptions, opt['batch_size'], shuffle=True), num_workers=opt['num_workers'])
    target_train_loader = DataLoader(PACSDatasetCLIP(target_train_examples, train_transform, descriptions), batch_sampler=ConditionalBatchSampler(target_train_examples, descriptions, opt['batch_size'], shuffle=True), num_workers=opt['num_workers'])
    source_val_loader = DataLoader(PACSDatasetBaseline(source_val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetBaseline(target_test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return source_train_loader, target_train_loader, source_val_loader, test_loader

