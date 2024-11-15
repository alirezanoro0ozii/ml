import random    
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import sys
import os
sys.path.append('../dlp')
sys.path.append('..')
from tqdm import tqdm
import gzip
import json
import os
import time
from pathlib import Path
from pprint import pprint
from urllib.parse import urlparse
import pyarrow.parquet as pq
import pickle

try:
    WORLD_SIZE=int(os.environ['WORLD_SIZE'])
    LOCAL_WORLD_SIZE=int(os.environ['LOCAL_WORLD_SIZE'])
    RANK=int(os.environ['RANK'])
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
except:
    WORLD_SIZE=1
    LOCAL_WORLD_SIZE=1
    LOCAL_RANK = 0
    RANK=0
print(  f' WORLD_SIZE={WORLD_SIZE} , LOCAL_WORLD_SIZE={LOCAL_WORLD_SIZE},RANK ={RANK},LOCAL_RANK = {LOCAL_RANK} ')

    
    
def gopen(path: str, mode: str):
    p = urlparse(path)
    if p.scheme == "gs":
        bucket = p.netloc
        return gcs_open(bucket, p.path, mode=mode)
    elif p.scheme == "":
        return open(p.path, mode=mode)
    else:
        assert False

def create_index(path:str, output_file: str):
    print('INDEXING DATASET')
    files_info_list = []
    files = os.listdir(path)
    files = [k for k in files if len(k)==12 and k.startswith('0000')]
    files = [str(Path(path) / f) for f in files]
    files.sort()  
    for uri in files:
        table = pq.read_table(uri)
        table_df = table.to_pandas()
        files_info_list.append({'uri':uri,'num_rows': len(table_df)})
    if not Path(output_file).is_file():
        with open(output_file, 'wb') as fp:
            pickle.dump(files_info_list, fp)

class PQDataAccess():
    
    def __init__(self, address,batch_size = 16, offset=None, world_size=None):
        
        self.address = address
        self.batch_size = batch_size
        if offset:
            self.offset = offset
        else :
            self.offset = RANK
        if world_size:
            self.world_size = world_size
        else :
            self.world_size = WORLD_SIZE
        self.p = address  
        self.iterator = None
        self.info_file= self.p+'_info'
        if not Path(self.info_file).is_file():
            create_index(self.p,self.info_file )
        self.cnt = 0
                
        with open (self.info_file, 'rb') as fp:
            self.files_info = pickle.load(fp)
            self.total_rows = sum([x['num_rows'] for x in self.files_info])
                        
    def get_item_with_start(self,start_index=None):
        if start_index== None:
            self.cnt = 0
            temp_list = []
            for file_index, info in enumerate(self.files_info):
                uri = info['uri']
                table = pq.read_table(uri)
                table_df = table.to_pandas()
                for index, row in table_df.iterrows():
                    if self.cnt %self.world_size == self.offset:
                        temp_list.append(row)
                        if len(temp_list) == self.batch_size:
                            new_list = list(temp_list)
                            temp_list=[]
                            
                            yield  new_list
                    self.cnt+=1
        else:   

            self.cnt = start_index
            temp_list = []

            # total_rows = sum([x['num_rows'] for x in self.files_info])
            start_index = start_index % self.total_rows

            n_rows,tot_rows,start_index_file = 0,0,0
            for file in self.files_info:
                n_rows+= file['num_rows']
                if n_rows > start_index:
                    break
                tot_rows= n_rows
                start_index_file+=1

            cur_rows = start_index - tot_rows

            for file_index, info in enumerate(self.files_info):
                if file_index < start_index_file:
                    continue
                else :
                    uri = info['uri']
                    table = pq.read_table(uri)
                    table_df = table.to_pandas()
                    if file_index > start_index_file:
                        cur_rows = 0
                    table_df_slice = table_df.iloc[cur_rows:]
                    for index, row in table_df_slice.iterrows():
                        if self.cnt %self.world_size == self.offset:
                            temp_list.append(row)
                            if len(temp_list) == self.batch_size:
                                new_list = list(temp_list)
                                temp_list=[]
                                
                                yield  new_list
                        self.cnt+=1
                        
    def create_iterator(self, start_index = None):
        if start_index == None:
            return self.get_item_with_start()
        else:
            return self.get_item_with_start(start_index)
        
    def set_iterator_index(self, start_index):
        self.cnt = start_index
        self.iterator = self.create_iterator(start_index)
    def get_batch(self):
        if self.iterator == None:
            self.iterator = self.create_iterator()
        try:
            elem = next(self.iterator)
        except StopIteration:
            self.iterator = self.create_iterator()
            elem = next(self.iterator)
        return elem