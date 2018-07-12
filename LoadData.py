import numpy as np
import os

class LoadData(object):
    '''given the path of data, return the data format for AFM and FM
    :param path
    return:
    Train_data: a dictionary, 'Y' refers to a list of y values; 'X' refers to a list of features_M dimension vectors with 0 or 1 entries
    Test_data: same as Train_data
    Validation_data: same as Train_data
    '''

    # Three files are needed in the path
    def __init__(self, path, dataset):
        self.path = path + "/"
        self.emgtrainfile = self.path +".emgtrain.txt"
        self.emgtestfile = self.path + ".emgtest.txt"
        self.imutrainfile = self.path +".imutrain.txt"
        self.imutestfile = self.path + ".imutest.txt"
        self.y_train = self.path + ".y_train.txt"
        self.y_test = self.path + ".y_test.txt"
        
        self.emgtraindata,self.emgtestdata,self.imutraindata,self.imutestdata,self.y_trainlabel,self.y_testlabel=self.read_data()



    def read_data(self):
        # read a data file. For a row, the first column goes into Y_;
        # the other columns become a row in X_ and entries are maped to indexs in self.features
        emgtr=self.load(self.emgtrainfile,64)
        emgte=self.load(self.emgtestfile,64)
        imutr=self.load(self.imutrainfile,26)
        imute=self.load(self.imutestfile,26)
        ytr=self.yload(self.y_train)
        yte=self.yload(self.y_est)
        return emgtr,emgte,imutr,imute,ytr,yte
    
    def load(self,file,num):
        f=open(file)
        X=[]
        fx=[]
        line = f.readline()
        while line:
            if len(line)<5:
                X.append(fx)
                fx=[]
            else:
                items = line.strip().split(' ')   
                fx.append( [ item for item in items] )
            line = f.readline()
        f.close()
        return X
    
    
        
        
