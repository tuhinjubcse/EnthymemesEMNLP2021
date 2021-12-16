import torch
from fairseq.models.bart import BARTModel
import os
import time
import numpy as np
datadir = 'enthymemes-paracomet1'
cpdir = 'PARACOMET1/'
bart = BARTModel.from_pretrained(cpdir,checkpoint_file='checkpoint_best.pt',data_name_or_path=datadir)

bart.cuda()
bart.eval()
np.random.seed(4)
torch.manual_seed(4)

count = 1
bsz = 1
maxb = 200
minb = 7

t = 0.7
elem = []
for val in [5]:
    with open('./ikatdataparacomet.source') as source, open('./ikatdataparacomet.hypo', 'w') as fout:
        sline = source.readline().strip()
        slines = [sline]
        for sline in source:
            if count % bsz == 0:
                with torch.no_grad():
                    hypotheses_batch = bart.sample(slines, beams=5, lenpen=2.0, max_len_b=maxb, min_len_b=minb, no_repeat_ngram_size=3)
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis.replace('\n','') + '\n')
                    fout.flush()
                slines = []

            slines.append(sline.strip())
            count += 1
        if slines != []:
            hypotheses_batch = bart.sample(slines, beams=5, lenpen=2.0, max_len_b=maxb, min_len_b=minb, no_repeat_ngram_size=3)
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis.replace('\n','') + '\n')
                fout.flush()