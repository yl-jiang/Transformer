#!/usr/bin/env python
# coding: utf-8

# In[1]:


from data import *
import sys
from pathlib import Path
from tqdm import tqdm


# In[2]:


cwd = Path('.').resolve()
sys.path.insert(0, str(cwd))


# In[3]:


if __name__ == "__main__":
    params = {'src_spacy_model_name':'zh_core_web_md', 
              'tgt_spacy_model_name':'en_core_web_md', 
              'src_txt_filepath':'./data/datasets/UN/en-zh/UNv1.0.en-zh.zh', 
              'tgt_txt_filepath':'./data/datasets/UN/en-zh/UNv1.0.en-zh.en', 
              'batch_size':10, 
              'num_workers':0, 
              'pin_memory': False, 
              'drop_last': True}
    
    
    def check_dataloader(pair, index=0):
        assert index < pair['src'].shape[0]
        ss_src, ss_tgt = "", ""
        
        for i in pair['src'].permute(1, 0)[index]:
            ss_src += collate.vocab_transform[collate.SRC_LANGUAGE].lookup_token(i) + " "

        for i in pair['trg'].permute(1, 0)[index]:
            ss_tgt += collate.vocab_transform[collate.TGT_LANGUAGE].lookup_token(i) + " "
        print(ss_src, ss_tgt, sep='\n')


    dataset, loader = build_wmt_dataloader(**params)
    collate = Collator(**params)
    for pair in tqdm(loader, total=len(loader)):
        # pair = next(iter(loader))
        check_dataloader(pair, 2)
        break


# In[1]:


# get_ipython().system('jupyter nbconvert --to script debug.py')


# In[ ]:




