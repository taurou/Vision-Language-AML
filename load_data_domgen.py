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


DG_labels = {"cartoon": {"art_painting": 0, "sketch": 1, "photo": 2},
             "sketch": {"art_painting": 0, "cartoon": 1, "photo": 2},
             "photo": {"art_painting": 0, "sketch": 1, "cartoon": 2}
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


class PACSDatasetBaseline_domgen(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        img_path, y, domain = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y, domain


def read_lines_domgen(data_path, target_domain):
    source_examples_per_domain = {}
    target_examples = {}

    for source_domain, _ in DG_labels[target_domain].items():
        source_examples = {}
        with open(f'{data_path}/{source_domain}.txt') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split()[0].split('/')
            category_name = line[3]
            category_idx = CATEGORIES[category_name]
            image_name = line[4]
            image_path = f'{data_path}/kfold/{source_domain}/{category_name}/{image_name}'
            if category_idx not in source_examples.keys():
                source_examples[category_idx] = [image_path]
            else:
                source_examples[category_idx].append(image_path)
            
        source_examples_per_domain[source_domain] = source_examples
    
    with open(f'{data_path}/{target_domain}.txt') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip().split()[0].split('/')
        category_name = line[3]
        category_idx = CATEGORIES[category_name]
        image_name = line[4]
        image_path = f'{data_path}/kfold/{target_domain}/{category_name}/{image_name}'
        if category_idx not in target_examples.keys():
            target_examples[category_idx] = [image_path]
        else:
            target_examples[category_idx].append(image_path)


    return source_examples_per_domain, target_examples


def build_splits_domgen(opt):
    target_domain = opt['target_domain']

    source_examples_per_domain, target_examples = read_lines_domgen(opt['data_path'], target_domain)
    
    train_examples = []
    val_examples = []
    test_examples = [] #target domain only

    # Compute ratios of examples for each category
    '''
    source_category_ratios_per_domain = {}
    source_total_examples_per_domain = {}
    val_split_length_per_domain = {}
    '''
    for source_domain, domain_idx in DG_labels[target_domain].items():
        '''
        source_category_ratios_per_domain[source_domain] = {category_idx: len(examples_list) for category_idx, examples_list in source_examples_per_domain[source_domain].items()}
        source_total_examples_per_domain[source_domain] = sum(source_category_ratios_per_domain[source_domain].values())
        source_category_ratios_per_domain[source_domain] = { category_idx: c / source_total_examples_per_domain[source_domain] for category_idx, c in source_examples_per_domain[source_domain].items()}

        # Build splits - we train only on the source domain (Art Painting)
        # 20% of the training split used for validation
        val_split_length_per_domain[source_domain] = { source_domain : total_examples * 0.2 for source_domain, total_examples in source_total_examples_per_domain.items() }
        '''
        source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples_per_domain[source_domain].items()}
        source_total_examples = sum(source_category_ratios.values())
        source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}

        # Build splits - we train only on the source domain (Art Painting)
        val_split_length = source_total_examples * 0.2 # 20% of the training split used for validation


        for category_idx, examples_list in source_examples_per_domain[source_domain].items():
            split_idx = round(source_category_ratios[category_idx] * val_split_length)
            for i, example in enumerate(examples_list):
                if i > split_idx:
                    # each pair is [path_to_img, class_label, domain_label]
                    train_examples.append([example, category_idx, domain_idx])
                else:
                    # each pair is [path_to_img, class_label, domain_label]
                    val_examples.append([example, category_idx, domain_idx])

    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            # each pair is [path_to_img, class_label]
            test_examples.append([example, category_idx, 3])

    random.seed(0)

    random.shuffle(train_examples) #shuffle because will be all elements belonging to a domain, then all the elements belonging to the other domain and so on...
    #random.shuffle(val_examples)


    # Transforms
    # ResNet18 - ImageNet Normalization
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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
    train_loader = DataLoader(PACSDatasetBaseline_domgen(train_examples, train_transform),batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetBaseline_domgen(val_examples, eval_transform),batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetBaseline_domgen(test_examples, eval_transform),batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader


class PACSDatasetCLIP_domgen(Dataset):
    def __init__(self, examples, transform, descr_dict):
        self.examples = examples
        self.transform = transform
        self.descr_dict = descr_dict

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        img_path, y, domain = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        key_dict = img_path.replace("data/PACS/kfold/", "")
        if key_dict in self.descr_dict:
            return x, y, domain, self.descr_dict[key_dict]
        else:
            return x, y, domain
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


def build_splits_clip_disentangle_domgen(opt):

    load_descriptions = json.load(
        open("./data/descriptions/all_labels_plus_assigned.json", "r"))
    descriptions = create_dict(load_descriptions)

    target_domain = opt['target_domain']

    source_examples_per_domain, target_examples = read_lines_domgen(opt['data_path'], target_domain)
    
    train_examples = []
    val_examples = []
    test_examples = [] #target domain only

    # Compute ratios of examples for each category
    '''
    source_category_ratios_per_domain = {}
    source_total_examples_per_domain = {}
    val_split_length_per_domain = {}
    '''
    for source_domain, domain_idx in DG_labels[target_domain].items():
        '''
        source_category_ratios_per_domain[source_domain] = {category_idx: len(examples_list) for category_idx, examples_list in source_examples_per_domain[source_domain].items()}
        source_total_examples_per_domain[source_domain] = sum(source_category_ratios_per_domain[source_domain].values())
        source_category_ratios_per_domain[source_domain] = { category_idx: c / source_total_examples_per_domain[source_domain] for category_idx, c in source_examples_per_domain[source_domain].items()}

        # Build splits - we train only on the source domain (Art Painting)
        # 20% of the training split used for validation
        val_split_length_per_domain[source_domain] = { source_domain : total_examples * 0.2 for source_domain, total_examples in source_total_examples_per_domain.items() }
        '''
        source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples_per_domain[source_domain].items()}
        source_total_examples = sum(source_category_ratios.values())
        source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}

        # Build splits - we train only on the source domain (Art Painting)
        val_split_length = source_total_examples * 0.2 # 20% of the training split used for validation


        for category_idx, examples_list in source_examples_per_domain[source_domain].items():
            split_idx = round(source_category_ratios[category_idx] * val_split_length)
            for i, example in enumerate(examples_list):
                if i > split_idx:
                    # each pair is [path_to_img, class_label, domain_label]
                    train_examples.append([example, category_idx, domain_idx])
                else:
                    # each pair is [path_to_img, class_label, domain_label]
                    val_examples.append([example, category_idx, domain_idx])

    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            # each pair is [path_to_img, class_label]
            test_examples.append([example, category_idx, 3])

    random.seed(0)
    random.shuffle(train_examples) #shuffle because will be all elements belonging to a domain, then all the elements belonging to the other domain and so on...
    #random.shuffle(val_examples)


    # Transforms
    # ResNet18 - ImageNet Normalization
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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
    train_loader = DataLoader(PACSDatasetCLIP_domgen(train_examples, train_transform, descriptions), batch_sampler=ConditionalBatchSampler(train_examples, descriptions, opt['batch_size'], shuffle=True), num_workers=opt['num_workers'])
    val_loader = DataLoader(PACSDatasetBaseline_domgen(val_examples, eval_transform),batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetBaseline_domgen(test_examples, eval_transform),batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader
#in target and validation we have the 3 source domains, in test we have the target domain only