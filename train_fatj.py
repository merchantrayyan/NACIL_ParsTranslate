import os
import codecs
from pydoc import doc
import shutil
import random
import numpy
import math
import torch
from sklearn.model_selection import train_test_split
import dp
from dp.utils.io import read_config, save_config
from dp.phonemizer import Phonemizer
from dp.preprocess import preprocess
from dp.train import train
from pyxdameraulevenshtein import damerau_levenshtein_distance, normalized_damerau_levenshtein_distance
import statistics
from pandas import DataFrame, read_csv
import pandas as pd
from subprocess import call

def get_data(iterate):
    #all this is the predetermined split
    train = []
    test = []
    dev = []

    with open("/home/rayyan/ParsTransliteration/training/datasets/"+str(iterate) + "_data/train.txt","r") as a:
        text = a.readlines()
        farsi = text[0::2]
        tajik = text[1::2]
        for i in range(0,len(tajik)):
            train.append(('fa',farsi[i].strip('\n'),tajik[i].strip('\n').lower()))
    with open("/home/rayyan/ParsTransliteration/training/datasets/"+str(iterate) + "_data/dev.txt","r") as a:
        text = a.readlines()
        farsi = text[0::2]
        tajik = text[1::2]
        for i in range(0,len(tajik)):
            dev.append(('fa',farsi[i].strip('\n'),tajik[i].strip('\n').lower()))
    with open("/home/rayyan/ParsTransliteration/training/datasets/"+str(iterate) + "_data/test.txt","r") as a:
        text = a.readlines()
        farsi = text[0::2]
        tajik = text[1::2]
        for i in range(0,len(tajik)):
            test.append(('fa',farsi[i].strip('\n'),tajik[i].strip('\n').lower()))
    #print(train)
    return train,test,dev

def write(phonemizer,test,filename):
    #farsi input test data
    #tajik output test data
    #test format: code, farsi, tajik
    predictions = []
    levens = []
    total = 0
    correct  = 0
    levD_one = 0
    levD_two = 0
    paired_dict = {}
    with open(filename,'w+') as y:
        y.write("Ansr\tAnsrXcrp\tRespXcrp\n")
    for line in test:
        phonemes = phonemizer(line[1],lang='fa')
        predictions.append(phonemes)
        #print(os.path.exists('testdata/predictions.txt'))
        #print("FAISUDHAOFIUSH")
        with codecs.open(filename,'a') as y:
            #whole doc is test set, FARSI, EXPECTED, PREDICTED, tab delimited
            y.write(line[1] + '\t' + line[2] + '\t' + phonemes + '\n')

def repeat(config):
    if(torch.cuda.device_count() !=1):
        train(config_file = config)
    else:
        train(rank=0,num_gpus=1,config_file=config)

def run(iterate: int):

    folder_freq = ("" + str(iterate)) + "_tests/"

    for i in range(0,iterate):

        train,test,dev = get_data(i)
        # EDIT CONFIG FILE
        # =============================================================
        config_file = 'fa_tj_config.yaml'
        config = read_config(config_file)
        config['model']['d_model'] = 256 #embedding dimension size
        config['model']['d_fft'] = 1024  #hidden layer size
        config['model']['layers'] = 4    ## of layers
        config['model']['dropout'] = 0.1 
        config['model']['heads'] = 4   
        config['preprocessing']['n_val'] = int(math.ceil((len(train)/9)))
        config['training']['learning_rate'] = 0.0005
        config['training']['batch_size'] = 16
        config['training']['batch_size_val'] = 16
        config['training']['epochs'] = 100
        
        BATCH_SIZE = 16
        compute_steps_per_epoch = lambda x: int(math.ceil(1. * x / BATCH_SIZE))
        steps_per_epoch = compute_steps_per_epoch((len(train)))    

        config['training']['generate_steps'] = steps_per_epoch*10
        config['training']['validate_steps'] = steps_per_epoch*10
        config['training']['warmup_steps'] = steps_per_epoch*5 #based on the dean of lismore transliteration paper, idk

        print(config)
        save_config(config,'current_config.yaml')

        # PREPROCESS & TRAIN W DP
        # =============================================================

        preprocess(config_file = 'current_config.yaml',train_data=train,val_data = dev)#oh shit why was i passing in the test data
        #may need to force train to repeat until it runs here due to that one error
        repeat('current_config.yaml')
        if os.path.exists("checkpoints/best_model.pt"):
            model = Phonemizer.from_checkpoint("checkpoints/best_model.pt")

            write(model,test,folder_freq + str(i)+"_100.txt")
 
            os.rename("checkpoints/best_model.pt","testdata/" + str(i) + "_best.pt")


        else:
            pass

        
        
run(10)