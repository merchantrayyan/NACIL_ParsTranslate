import os
import pandas as pd
from pyxdameraulevenshtein import damerau_levenshtein_distance, normalized_damerau_levenshtein_distance
import statistics
from pandas import DataFrame, read_csv
import numpy
from collections import Counter
import numpy as np
import difflib
from sacrebleu.metrics import BLEU, CHRF, TER
import subprocess
# folder_names = []
#folder_names = []
folder_names = ["10_folds"]
paired_dict = dict()

#iteration referred to the models trained with [iteration] number of epochs
def calculate(folder,iteration,dataframe,dictionary):
    files = os.listdir(folder)
    for file in files:
        #rint(file)
        if file.find("_"+str(iteration) + '.txt') != -1:
            print(file)
            data_df = pd.read_csv(folder + "/" + file,delimiter='\t',on_bad_lines='skip')
            #print(data_df)
            test = data_df.loc[:,"AnsrXcrp"]
            predictions = data_df.loc[:,'RespXcrp']
            inputs = data_df.loc[:,'Ansr']

            refs = []
            refset = list()
            for sentence in test:
                if not isinstance(sentence,str):
                    sentence = ""
                    #print(len(sentence))
                #sentence = sentence.replace('‌','')
                    
                    #print(len(sentence.replace('‌','')))
                    #print(sentence
                refset.append(sentence)
            #print(refset)
            refs.append(refset)
            with open("references.txt",'w+') as a:
                for item in refset:
                    a.write(item+'\n')
            refset = list()
            for sentence in predictions:
                if not isinstance(sentence,str):
                    sentence = ""
                    #print(sentence)
                #sentence = sentence.replace('‌','')
                    #print(sentence)
                refset.append(sentence)
            #print((refset))
            sys = refset

            with open("predictions.txt",'w+') as a:
                for item in refset:
                    a.write(item+'\n')

            subprocess.call('python3 chrF++.py -R references.txt -H predictions.txt > score.txt',shell=True)
            with open("score.txt",'r') as a:
                line = a.readlines()
                scores = line[2].split('\t')
                chrfplus = scores[1].strip('\n')
                print(chrfplus)

            bleu = BLEU()
            chrf = CHRF()
            bleus = bleu.corpus_score(sys, refs)
            charf = chrf.corpus_score(sys, refs)


            levens = []
            lds = []
            total = 0
            correct  = 0
            levD_one = 0
            levD_two = 0
            same_wc = 0
            seq_acc = 0
            
            ld_nums = []
            norm_nums = []
            seq1 = 0
            seq2= 0
            seq3 = 0
            seq4 = 0
            seq5 = 0
            zwnj = 0
            zwnjseq = 0
            for x in range(0,len(test)):
                    #split the prediction and expectation by space
                    #print(predictions)
                    try:
                        pred = predictions[x]
                        pred = pred.replace('‌','')
                        prediction_split = pred.split(' ')
                        line = test[x]
                        line = line.replace('‌','')

                        if not isinstance(line,str):
                            line = ""
                        #print(line)
                        expected_split = line.split(' ')
                        #print(expected_split + " ads")
                        diff = abs((len(prediction_split) - len(expected_split)))
                    except AttributeError:
                        print(test[x] + "///" + str(predictions[x]))
                        diff = -1
                    if diff == 0:
                        same_wc += 1
                        inputted = inputs[x]
                        total += len(prediction_split)
                        for y in range(0, len(expected_split)):
                            #print(y)
                            distance = damerau_levenshtein_distance(prediction_split[y],expected_split[y])
                            if distance == 0:
                                correct += 1
                            elif distance == 1:
                                levD_one += 1
                                #if expected_split[y].find('‌') != -1 and prediction_split[y].find('‌') == -1:
                                    #zwnj += 1
                                #if expected_split[y].find('ъ') != -1 and prediction_split[y].find('ъ') == -1:
                                    #zwnj += 1

                            elif distance == 2: 
                                levD_two += 1
                            levD_norm = normalized_damerau_levenshtein_distance(prediction_split[y],expected_split[y])
                            levens.append(levD_norm)
                            lds.append(distance)
                            if(y < len(inputted)):
                                if(inputted[y] not in paired_dict):
                                    paired_dict[inputted[y]] = {'L2_actual': expected_split[y], 
                                    'L2_predict': prediction_split[y],
                                    'freq': 1,'Lev': distance, 
                                    'Lev_N':round(levD_norm,2)}
                                else:
                                    paired_dict[inputted[y]]['freq'] = paired_dict[inputted[y]]['freq'] + 1

                    prediction_mish = ''.join(prediction_split)
                    expected_mish = ''.join(expected_split)
                    ld = damerau_levenshtein_distance(prediction_mish,expected_mish)
                    if (ld == 0):
                        seq_acc =  seq_acc + 1
                    elif (ld ==1):
                        seq1 += 1
                    elif (ld ==2):
                        seq2 += 1
                    elif (ld ==3):
                        seq3 += 1
                    elif (ld ==4):
                        seq4 += 1
                    elif (ld ==5):
                        seq5 += 1
                    mish = prediction_mish.replace('‌','')
                    emish = expected_mish.replace('‌','')
                    newld = damerau_levenshtein_distance(mish,emish)
                    if newld == 0:
                        zwnjseq += 1

                    ld_nums.append(ld)
                    norm_nums.append(normalized_damerau_levenshtein_distance(prediction_mish,expected_mish))

            norm_distance = numpy.sum(levens)/(len(levens)) #for all the seqs with same number of words, what is the avg NLD for that
            first = correct/total
            second = ((correct + levD_one)/total)
            second_ezafe = (correct + zwnj)/total
            third = ((correct + levD_one+levD_two)/total)
            seqs = round(seq_acc/len(test),3)
            seqs1u = round((seq_acc + seq1)/len(test),3)
            seqs2u = round((seq_acc + seq2 + seq1)/len(test),3)
            seqs3u = round((seq_acc + seq3+ seq2 + seq1)/len(test),3)
            seqs4u = round((seq_acc + seq4+ seq3+ seq2 + seq1)/len(test),3)
            seqs5u = round((seq_acc + seq5+seq4+ seq3+ seq2 + seq1)/len(test),3)
            dataframe.loc[len(dataframe.index)] = [round(first,5),round(second,5),round(second_ezafe,5),round(third,5),round(statistics.mean(lds),3),round(norm_distance,3),round(statistics.mean(ld_nums),3),round(statistics.mean(norm_nums),3),round((same_wc/len(test)),3),seqs,seqs1u,seqs2u,seqs3u,seqs4u,seqs5u,zwnjseq/len(test),bleus,charf,chrfplus]
    
    dataframe.to_csv("/ParsTransliteration/measurement/" + folder +"_"+str(iteration)+ ".csv",index=False,header=True)
        
def run(iteration):
    for folder in folder_names:
        print(folder)
        df = pd.DataFrame(columns=['0_ED','1_ED',"1_ZWNJ",'2_ED','AVG_LD','AVG_NLD','SEQ_AVG_LD','SEQAVG_NLD','WC_SAME','SEQ_ACC','SLD1','SLD2','SL31','SLD4','SLD5','ZWNJ1','BLEU','CHRF','CHRF++'])
        calculate(folder,iteration,df,paired_dict)
        df_freq = pd.DataFrame(columns=['L1_actual','L2_actual','L2_predict','freq',"Lev","Lev_N"])
        for one,val in sorted(paired_dict.items()):
            df_freq.loc[len(df_freq.index)] = [(one),paired_dict[one]['L2_actual'],paired_dict[one]['L2_predict'] ,paired_dict[one]['freq'] ,paired_dict[one]['Lev'], paired_dict[one]['Lev_N']]
        df_freq.to_csv("paired_freq.csv")

run(100)

