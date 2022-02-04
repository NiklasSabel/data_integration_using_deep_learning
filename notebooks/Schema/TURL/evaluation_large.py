from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import pandas as pd
from IPython.display import display

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import argparse

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import trange
from tqdm.autonotebook import tqdm

from data_loader.hybrid_data_loaders import *
from data_loader.header_data_loaders import *
from data_loader.CT_Wiki_data_loaders import *
#from data_loader.RE_data_loaders import *
#from data_loader.EL_data_loaders import *
from model.configuration import TableConfig
from model.model import HybridTableMaskedLM, HybridTableCER, TableHeaderRanking, HybridTableCT,HybridTableEL,HybridTableRE,BertRE
from model.transformers import BertConfig,BertTokenizer, WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from utils.util import *
from baselines.row_population.metric import average_precision,ndcg_at_k
from baselines.cell_filling.cell_filling import *
from model import metric

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'CER': (TableConfig, HybridTableCER, BertTokenizer),
    'CF' : (TableConfig, HybridTableMaskedLM, BertTokenizer),
    'HR': (TableConfig, TableHeaderRanking, BertTokenizer),
    'CT': (TableConfig, HybridTableCT, BertTokenizer),
    'EL': (TableConfig, HybridTableEL, BertTokenizer),
    'RE': (TableConfig, HybridTableRE, BertTokenizer),
    'REBERT': (BertConfig, BertRE, BertTokenizer)
}

# set data directory, this will be used to load test data
data_dir =  '/pfs/work7/workspace/scratch/ma_luitheob-tp2021/turl/turl-base/large/input/'

config_name = "configs/table-base-config_v2.json"
device = torch.device('cuda')
# load entity vocab from entity_vocab.txt
entity_vocab = load_entity_vocab(data_dir, ignore_bad_title=True, min_ent_count=2)
entity_wikid2id = {entity_vocab[x]['wiki_id']:x for x in entity_vocab}

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


with open(os.path.join(data_dir, 'test_representation.txt'), 'rb') as f:
    example = pickle.load(f)[0]
display(example)

# load type vocab from type_vocab.txt
type_vocab = load_type_vocab(data_dir)
test_dataset = WikiCTDataset(data_dir, entity_vocab, type_vocab, max_input_tok=500, src="test", max_length = [50, 10, 10], force_new=False, tokenizer = None)
len(type_vocab)

id2type = {idx:t for t, idx in type_vocab.items()}
t2d_invalid = set()

def average_precision(output, relevance_labels):
    with torch.no_grad():
        sorted_output = torch.argsort(output, dim=-1, descending=True)
        sorted_labels = torch.gather(relevance_labels, -1, sorted_output).float()
        cum_correct = torch.cumsum(sorted_labels, dim=-1)
        cum_precision = cum_correct / torch.arange(start=1,end=cum_correct.shape[-1]+1, device=cum_correct.device)[None, :]
        cum_precision = cum_precision * sorted_labels
        total_valid = torch.sum(sorted_labels, dim=-1)
        total_valid[total_valid==0] = 1
        average_precision = torch.sum(cum_precision, dim=-1)/total_valid
    return average_precision

per_type_accuracy = {}
per_type_precision = {}
per_type_recall = {}
per_type_f1 = {}
map = {}
precision = {}
recall = {}
f1 = {}
per_table_result = {}

from tqdm.autonotebook import tqdm
checkpoints = [
    "large/output/'mode_$'/pytorch_model.bin",
    "large/output/mode_1/pytorch_model.bin",
    "large/output/mode_2/pytorch_model.bin",
    "large/output/mode_3/pytorch_model.bin",
    "large/output/mode_4/pytorch_model.bin"
]

mode = 0
#for mode in range(4):
print(mode)
config_class, model_class, _ = MODEL_CLASSES['CT']
config = config_class.from_pretrained(config_name)
config.class_num = len(type_vocab)
config.mode = mode
model = model_class(config, is_simple=True)
checkpoint = checkpoints[mode]
checkpoint = torch.load(checkpoint)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()
eval_batch_size = 20
eval_sampler = SequentialSampler(test_dataset)
eval_dataloader = CTLoader(test_dataset, sampler=eval_sampler, batch_size=eval_batch_size, is_train=False)
eval_loss = 0.0
eval_map = 0.0
nb_eval_steps = 0
eval_targets = []
eval_prediction_scores = []
eval_pred = []
eval_mask = []
per_table_result[mode] = {}
for batch in tqdm(eval_dataloader, desc="Evaluating"):
    table_ids, input_tok, input_tok_type, input_tok_pos, input_tok_mask, \
        input_ent_text, input_ent_text_length, input_ent, input_ent_type, input_ent_mask, \
        column_entity_mask, column_header_mask, labels_mask, labels = batch
    input_tok = input_tok.to(device)
    input_tok_type = input_tok_type.to(device)
    input_tok_pos = input_tok_pos.to(device)
    input_tok_mask = input_tok_mask.to(device)
    input_ent_text = input_ent_text.to(device)
    input_ent_text_length = input_ent_text_length.to(device)
    input_ent = input_ent.to(device)
    input_ent_type = input_ent_type.to(device)
    input_ent_mask = input_ent_mask.to(device)
    column_entity_mask = column_entity_mask.to(device) 
    column_header_mask = column_header_mask.to(device)
    labels_mask = labels_mask.to(device)
    labels = labels.to(device)
    if mode == 1:
        input_ent_mask = input_ent_mask[:,:,input_tok_mask.shape[1]:]
        input_tok = None
        input_tok_type = None
        input_tok_pos = None
        input_tok_mask = None
    elif mode == 2:
        input_tok_mask = input_tok_mask[:,:,:input_tok_mask.shape[1]]
        input_ent_text = None
        input_ent_text_length = None
        input_ent = None
        input_ent_type = None
        input_ent_mask = None
    elif mode == 3:
        input_ent = None
    elif mode == 4:
        input_ent_mask = input_ent_mask[:,:,input_tok_mask.shape[1]:]
        input_tok = None
        input_tok_type = None
        input_tok_pos = None
        input_tok_mask = None
        input_ent = None
    elif mode == 5:
        input_ent_mask = input_ent_mask[:,:,input_tok_mask.shape[1]:]
        input_tok = None
        input_tok_type = None
        input_tok_pos = None
        input_tok_mask = None
        input_ent_text = None
        input_ent_text_length = None
    with torch.no_grad():
        outputs = model(input_tok, input_tok_type, input_tok_pos, input_tok_mask,\
            input_ent_text, input_ent_text_length, input_ent, input_ent_type, input_ent_mask, column_entity_mask, column_header_mask, labels_mask, labels)
        loss = outputs[0]
        prediction_scores = outputs[1]
        for l_i in t2d_invalid:
            prediction_scores[:,:,l_i] = -1000
        for idx, table_id in enumerate(table_ids):
            valid = labels_mask[idx].nonzero().max().item()+1
            if table_id not in per_table_result[mode]:
                per_table_result[mode][table_id] = [[],labels_mask[idx,:valid],labels[idx,:valid]]
            per_table_result[mode][table_id][0].append(prediction_scores[idx,:valid])
        ap = metric.average_precision(prediction_scores.view(-1, config.class_num), labels.view((-1, config.class_num)))
        map = (ap*labels_mask.view(-1)).sum()/labels_mask.sum()
        eval_loss += loss.mean().item()
        eval_map += map.item()
        eval_targets.extend(labels.view(-1, config.class_num).tolist())
        eval_prediction_scores.extend(prediction_scores.view(-1, config.class_num).tolist())
        eval_pred.extend((prediction_scores.view(-1, config.class_num)==prediction_scores.view(-1, config.class_num).max(-1)[0][:,None]).tolist())
        eval_mask.extend(labels_mask.view(-1).tolist())
    nb_eval_steps += 1
print(eval_map/nb_eval_steps)
eval_targets = np.array(eval_targets)
eval_prediction_scores = np.array(eval_prediction_scores)
eval_mask = np.array(eval_mask)
eval_prediction_ranks = np.argsort(np.argsort(-eval_prediction_scores))
eval_pred = np.array(eval_pred)
eval_tp = eval_mask[:,np.newaxis]*eval_pred*eval_targets
eval_precision = np.sum(eval_tp,axis=0)/np.sum(eval_mask[:,np.newaxis]*eval_pred,axis=0)
eval_precision = np.nan_to_num(eval_precision, 1)
eval_recall = np.sum(eval_tp,axis=0)/np.sum(eval_mask[:,np.newaxis]*eval_targets,axis=0)
eval_recall = np.nan_to_num(eval_recall, 1)
eval_f1 = 2*eval_precision*eval_recall/(eval_precision+eval_recall)
eval_f1 = np.nan_to_num(eval_f1, 0)
per_type_instance_num = np.sum(eval_mask[:,np.newaxis]*eval_targets,axis=0)
per_type_instance_num[per_type_instance_num==0] = 1
per_type_correct_instance_num = np.sum(eval_mask[:,np.newaxis]*(eval_prediction_ranks<eval_targets.sum(axis=1)[:,np.newaxis])*eval_targets,axis=0)
per_type_accuracy[mode] = per_type_correct_instance_num/per_type_instance_num
per_type_precision[mode] = eval_precision
per_type_recall[mode] = eval_recall
per_type_f1[mode] = eval_f1
precision[mode] = np.sum(eval_tp)/np.sum(eval_mask[:,np.newaxis]*eval_pred)
recall[mode] = np.sum(eval_tp)/np.sum(eval_mask[:,np.newaxis]*eval_targets)
f1[mode] = 2*precision[mode]*recall[mode]/(precision[mode]+recall[mode])


total_corr = 0
total_valid = 0
errors = []
for table_id, result in per_table_result[mode].items():
    prediction_scores, label_mask, label = result
    prediction_scores = torch.stack(prediction_scores, 0).mean(0)
    current_corr = 0
    for col_idx, pred in enumerate(prediction_scores.argmax(-1).tolist()):
        current_corr += label[col_idx, pred].item()
    total_valid += label_mask.sum().item()
    total_corr += current_corr
    if current_corr!=label_mask.sum().item():
        errors.append(table_id)
print(total_corr/total_valid, total_valid)

#for t,i in sorted(type_vocab.items(),key=lambda z:-per_type_instance_num[z[1]]):
#    print('%s %.4f %.4f %.4f %.4f %.4f  %.4f %.4f'%(t, per_type_instance_num[i], per_type_f1[0][i], per_type_f1[4][i], per_type_f1[1][i], per_type_f1[3][i], per_type_f1[2][i], per_type_f1[5][i]))
#    print()

target_list = ['Product.name',
 'Product.offers',
 'Product.brand',
 'CreativeWork.headline',
 'CreativeWork.author',
 'CreativeWork.datepublished',
 'Product.aggregaterating',
 'CreativeWork.datemodified',
 'Product.mpn',
 'Product.category',
 'Event.name',
 'Event.enddate',
 'Event.location',
 'Product.review',
 'MusicRecording.name',
 'MusicRecording.duration',
 'MusicRecording.byartist',
 'LocalBusiness.name',
 'Event.performer',
 'LocalBusiness.address',
 'Product.manufacturer',
 'MusicRecording.inalbum',
 'Product.itemcondition',
 'Recipe.name',
 'Product.model',
 'LocalBusiness.telephone',
 'CreativeWork.name',
 'Recipe.recipeinstructions',
 'Product.weight',
 'Recipe.author',
 'CreativeWork.publisher',
 'Recipe.recipeyield',
 'Event.organizer',
 'Product.price',
 'Recipe.recipeingredient',
 'Person.mainentityofpage',
 'Recipe.preptime',
 'Product.gtin13',
 'Recipe.cooktime',
 'Recipe.datepublished',
 'Product.releasedate',
 'Recipe.totaltime',
 'Recipe.recipecategory',
 'CreativeWork.mainentityofpage',
 'Place.name',
 'Product.color',
 'Event.eventstatus',
 'Recipe.recipecuisine',
 'Recipe.aggregaterating',
 'Recipe.keywords',
 'LocalBusiness.geo',
 'Place.geo',
 'Recipe.nutrition',
 'LocalBusiness.pricerange',
 'CreativeWork.interactionstatistic',
 'Recipe.mainentityofpage',
 'Recipe.ispartof',
 'Product.mainentityofpage',
 'Product.isrelatedto',
 'Product.gtin12',
 'LocalBusiness.aggregaterating',
 'LocalBusiness.email',
 'Event.eventattendancemode',
 'Product.pricecurrency',
 'Recipe.ingredients',
 'Product.width',
 'CreativeWork.datecreated',
 'Product.height',
 'Book.name',
 'Product.additionalproperty',
 'Person.jobtitle',
 'Product.depth',
 'Place.telephone',
 'Product.gtin8',
 'Person.birthdate',
 'Person.givenname',
 'Person.familyname',
 'Person.telephone',
 'Book.author',
 'LocalBusiness.sameas',
 'Person.email',
 'LocalBusiness.faxnumber',
 'Product.gtin',
 'LocalBusiness.openinghours',
 'Restaurant.name',
 'Hotel.name',
 'Person.deathdate',
 'Book.isbn',
 'Restaurant.address',
 'Person.additionalname',
 'Hotel.address',
 'Book.publisher',
 'Restaurant.telephone',
 'Book.offers',
 'Book.datepublished',
 'LocalBusiness.openinghoursspecification',
 'CreativeWork.alternativeheadline',
 'Product.identifier',
 'Product.availability',
 'Person.gender',
 'Restaurant.servescuisine',
 'Hotel.telephone',
 'Product.gtin14',
 'Event.duration',
 'Product.shop-currency',
 'Restaurant.geo',
 'Book.numberofpages',
 'Restaurant.pricerange',
 'Recipe.performtime',
 'CreativeWork.keywords',
 'LocalBusiness.review',
 'CreativeWork.articlebody',
 'Place.sameas',
 'LocalBusiness.contactpoint',
 'Product.reviewcount',
 'Person.worksfor',
 'Hotel.pricerange',
 'Person.height',
 'Hotel.geo',
 'Book.bookformat',
 'Person.address',
 'CreativeWork.aggregaterating',
 'Person.nationality',
 'Product.ratingvalue',
 'Hotel.aggregaterating',
 'Product.issimilarto',
 'Person.weight',
 'Person.affiliation',
 'Restaurant.aggregaterating',
 'Person.homelocation',
 'Product.itemlistelement',
 'CreativeWork.interactioncount',
 'CreativeWork.inlanguage',
 'Product.bestrating',
 'Book.inlanguage',
 'Recipe.cookingmethod',
 'LocalBusiness.legalname',
 'Recipe.review',
 'CreativeWork.commentcount',
 'Recipe.suitablefordiet',
 'Restaurant.hasmenu',
 'Restaurant.acceptsreservations',
 'Person.knowslanguage',
 'Restaurant.openinghoursspecification',
 'Book.genre',
 'CreativeWork.interactiontype',
 'Book.aggregaterating',
 'Product.condition',
 'MusicAlbum.name',
 'Person.makesoffer',
 'Product.worstrating',
 'Product.alternatename',
 'LocalBusiness.areaserved',
 'LocalBusiness.location',
 'LocalBusiness.addresslocality',
 'CreativeWork.creator',
 'Product.audience',
 'Person.sameas',
 'LocalBusiness.addressregion',
 'LocalBusiness.streetaddress',
 'LocalBusiness.postalcode',
 'TVEpisode.name',
 'Product.breadcrumb',
 'Product.material',
 'Product.disambiguatingdescription',
 'MusicRecording.datepublished',
 'Hotel.starrating',
 'LocalBusiness.paymentaccepted',
 'Product.title',
 'Product.address',
 'CreativeWork.genre',
 'Book.bookedition',
 'Recipe.datemodified',
 'LocalBusiness.founder',
 'LocalBusiness.citystatezip',
 'MusicAlbum.byartist',
 'CreativeWork.version',
 'Event.doortime',
 'Place.openinghoursspecification',
 'Product.author',
 'Recipe.publisher',
 'MusicRecording.genre',
 'Event.typicalagerange',
 'Restaurant.openinghours',
 'MusicRecording.datecreated',
 'CreativeWork.offers',
 'Event.performers',
 'TVEpisode.partofseries',
 'Person.birthplace',
 'Product.offerdetails',
 'MusicAlbum.track',
 'Person.memberof',
 'LocalBusiness.owns',
 'Product.reviews',
 'Person.faxnumber',
 'CreativeWork.copyrightholder',
 'Recipe.headline',
 'CreativeWork.copyrightyear',
 'LocalBusiness.hasmap',
 'Product.geo',
 'Person.worklocation',
 'Product.datepublished',
 'MusicAlbum.numtracks',
 'Restaurant.menu',
 'Person.contactpoint',
 'Hotel.email',
 'TVEpisode.episodenumber']

for t,i in sorted(type_vocab.items(),key=lambda z:-per_type_instance_num[z[1]]):
    if t in target_list:
        print('%s %.4f %.4f'%(t, per_type_instance_num[i], per_type_f1[mode][i]))
        print()

print('F1 Score:')
print(f1)
print('Precision:')
print(precision)
print('Recall')
print(recall)
print('Errors:')
print(errors)