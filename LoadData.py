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
    def __init__(self, path):
        self.path = path + "/"
        self.emgtrainleftfile = self.path +"emgtrainleft.txt"
        self.emgtestleftfile = self.path + "emgtestleft.txt"
        self.acctrainleftfile = self.path +"acctrainleft.txt"
        self.acctestleftfile = self.path + "acctestleft.txt"
        self.emgtrainrightfile = self.path +"emgtrainright.txt"
        self.emgtestrightfile = self.path + "emgtestright.txt"
        self.acctrainrightfile = self.path +"acctrainright.txt"
        self.acctestrightfile = self.path + "acctestright.txt"
        self.gyrtrainleftfile = self.path +"gyrtrainleft.txt"
        self.gyrtestleftfile = self.path + "gyrtestleft.txt"
        self.gyrtrainrightfile = self.path +"gyrtrainright.txt"
        self.gyrtestrightfile = self.path + "gyrtestright.txt"
        self.oltrainleftfile = self.path +"oltrainleft.txt"
        self.oltestleftfile = self.path + "oltestleft.txt"
        self.oltrainrightfile = self.path +"oltrainright.txt"
        self.oltestrightfile = self.path + "oltestright.txt"
        self.oritrainleftfile = self.path +"oritrainleft.txt"
        self.oritestleftfile = self.path + "oritestleft.txt"
        self.oritrainrightfile = self.path +"oritrainright.txt"
        self.oritestrightfile = self.path + "oritestright.txt"
        self.y_train = self.path + "y_train.txt"
        self.y_test = self.path + "y_ctest.txt"
        self.cnn_label = self.path + "cnn_y_train.txt"
        [self.emgtrainleft,self.emgtestleft ,self.emgtrainright ,self.emgtestright ,
         self.acctrainleft,self.acctestleft ,self.acctrainright ,self.acctestright,
        self.gyrtrainleft , self.gyrtestleft ,self.gyrtrainright ,self.gyrtestright ,
        self.oltrainleft ,self.oltestleft ,self.oltrainright ,self.oltestright ,
        self.oritrainleft ,self.oritestleft ,self.oritrainright,self.oritestright ,self.ytrain,
        self.ytest,self.ohytr,self.tr_len,self.ohyte,self.te_len ,self.cnn]=self.read_data()

        self.emgctestleftfile = self.path + "emgctestleft.txt"
        self.emgctestrightfile = self.path + "emgctestright.txt"
        self.accctestleftfile = self.path + "accctestleft.txt"
        self.accctestrightfile = self.path + "accctestright.txt"
        self.gyrctestleftfile = self.path + "gyrctestleft.txt"
        self.gyrctestrightfile = self.path + "gyrctestright.txt"
        self.orictestleftfile = self.path + "orictestleft.txt"
        self.orictestrightfile = self.path + "orictestright.txt"
        self.olctestleftfile = self.path + "olctestleft.txt"
        self.olctestrightfile = self.path + "olctestright.txt"
        


    def cdata(self):
        etl=self.load(self.emgctestleftfile)    
        atl=self.load(self.accctestleftfile)
        etr=self.load(self.emgctestrightfile)
        atr=self.load(self.accctestrightfile)
        gtl=self.load(self.gyrctestleftfile)
        gtr=self.load(self.gyrctestrightfile)
        ltl=self.load(self.olctestleftfile)
        ltr=self.load(self.olctestrightfile)
        otl=self.load(self.orictestleftfile)
        otr=self.load(self.orictestrightfile)
        return etl,etr,atl,atr,gtl,gtr,ltl,ltr,otl,otr

    def read_data(self):
        # read a data file. For a row, the first column goes into Y_;
        # the other columns become a row in X_ and entries are maped to indexs in self.features
        enl=self.load(self.emgtrainleftfile)
        etl=self.load(self.emgtestleftfile) 
        anl=self.load(self.acctrainleftfile)     
        atl=self.load(self.acctestleftfile)
        enr=self.load(self.emgtrainrightfile)
        etr=self.load(self.emgtestrightfile)
        anr=self.load(self.acctrainrightfile)
        atr=self.load(self.acctestrightfile)
        gnl=self.load(self.gyrtrainleftfile)
        gtl=self.load(self.gyrtestleftfile)
        gnr=self.load(self.gyrtrainrightfile)
        gtr=self.load(self.gyrtestrightfile)
        lnl=self.load(self.oltrainleftfile)
        ltl=self.load(self.oltestleftfile)
        lnr=self.load(self.oltrainrightfile)
        ltr=self.load(self.oltestrightfile)
        onl=self.load(self.oritrainleftfile)
        otl=self.load(self.oritestleftfile)
        onr=self.load(self.oritrainrightfile)
        otr=self.load(self.oritestrightfile)
        ytr,tr_len=self.yload(self.y_train)
        yte,te_len=self.yload(self.y_test)
        ohytr=self.yyload(self.y_train)
        ohyte=self.yyload(self.y_test)
        cnn=self.cnn_load(self.cnn_label)
        return enl,etl,enr,etr,anl,atl,anr,atr,gnl,gtl,gnr,gtr,lnl,ltl,lnr,ltr,onl,otl,onr,otr,ytr,yte,ohytr,tr_len,ohyte,te_len,cnn
    
    
    def load(self,file):
        f=open(file)
        X=[]
        fx=[]

        line = f.readline()
        while line:
            items = line.strip().split(' ')   
            if len(line)<5:
                
                X.append(fx)
                fx=[]
            else:
                fx.append( [ [float(item)] for item in items] )
            line = f.readline()
        f.close()
        return X
        
    def cnn_load(self,file):
        f=open(file)
        Y=[]
        line = f.readline()
        while line:
            if len(line)>0:
                Y.append(int(line)-1)
            line = f.readline()
        f.close()
        return np.eye(30)[np.array(Y, dtype=np.int32)]

    def yload(self,file):
        f=open(file)
        Y=[]
        y_len=[]
        line = f.readline()
        while line:
            items = line.strip().split(' ')
            items.insert(0, '0')
            t=len(items)
            while items[t-1]=='0':
                t=t-1
            Y.append( [ int(item) for item in items] )
            y_len.append(t)
            line = f.readline()
        f.close
        return Y,y_len
    
    def yyload(self,file):
        f=open(file)
        Y=[]
        line = f.readline()
        while line:
            items = line.strip().split(' ')
            items.insert(0, '0')
            Y.append( np.eye(36)[np.array(items, dtype=np.int32)] )
            line = f.readline()
        f.close
        return Y

    def getdata(self):
        return         [self.emgtrainleft, self.emgtestleft , self.emgtrainright ,self.emgtestright ,
                        self.acctrainleft,self.acctestleft ,self.acctrainright,self.acctestright ,
        self.gyrtrainleft ,self.gyrtestleft ,self.gyrtrainright ,self.gyrtestright ,
        self.oltrainleft ,self.oltestleft ,self.oltrainright , self.oltestright ,
        self.oritrainleft , self.oritestleft ,self.oritrainright, self.oritestright ,
        self.ytrain,self.ytest,self.ohytr,self.tr_len,self.ohyte,self.te_len]

    def getcnn(self):
        return self.cnn
        
    def gettrainlabel(self):
        return self.y_trainlabel
        
    def gettestlabel(self):
        return self.y_testlabel    