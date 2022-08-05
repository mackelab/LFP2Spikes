import pandas as pd
import numpy as np
from database.db_setup import *
from preprocessing.data_preprocessing.binning import *
from random import randint


class LOADER(): 
    
    def __init__(self): 
        
        self.patient_ids = [66]
        print(f"Default patient IDs : {self.get_patient_ids()}")

        
        
    def get_patient_ids(self): 
        return self.patient_ids
    
    
    
    