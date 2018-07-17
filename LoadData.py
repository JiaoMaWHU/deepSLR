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
        self.y_test = self.path + "y_test.txt"
        
        [self.emgtrainleft,self.emgtestleft ,self.emgtrainright ,self.emgtestright ,
         self.acctrainleft,self.acctestleft ,self.acctrainright ,self.acctestright,
        self.gyrtrainleft , self.gyrtestleft ,self.gyrtrainright ,self.gyrtestright ,
        self.oltrainleft ,self.oltestleft ,self.oltrainright ,self.oltestright ,
        self.oritrainleft ,self.oritestleft ,self.oritrainright,self.oritestright ,self.ytrain,
        self.ytest,self.ohytr,self.tr_len,self.ohyte,self.te_len ]=self.read_data()


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
        ytr=self.yload(self.y_train)
        yte=self.yload(self.y_test)
        ohytr,tr_len=self.yyload(self.y_train)
        ohyte,te_len=self.yyload(self.y_test)
        return enl,etl,enr,etr,anl,atl,anr,atr,gnl,gtl,gnr,gtr,lnl,ltl,lnr,ltr,onl,otl,onr,otr,ytr,yte,ohytr,tr_len,ohyte,te_len
    
    
    def load(self,file):
        f=open(file)
        X=[]
        fx=[]
        a=[]
        t=0
        num=0
        line = f.readline()
        while line:
            items = line.strip().split(' ')   
            if len(line)<5:
                while num<402:
                    fx.append(a)
                    num=num+1
                num=0
                X.append(fx)
                fx=[]
            else:
                a=[]
                t=len(items)
                while t>0:
                    a.append(0)  
                    t=t-1
                fx.append( [ float(item) for item in items] )
                num=num+1
            line = f.readline()
        f.close()
        return X
    
    def yload(self,file):
        f=open(file)
        Y=[]
        line = f.readline()
        while line:
            items = line.strip().split(' ')
            t=len(items)
            while t<8:
                line=line+'0 '
                t=t+1
            items = line.strip().split(' ')    
            Y.append( [ int(item) for item in items] )
            line = f.readline()
        f.close
        return Y
    
    def yyload(self,file):
        f=open(file)
        Y=[]
        y_len=[]
        line = f.readline()
        while line:
            items = line.strip().split(' ')
            t=len(items)
            while t<8:
                line=line+'0 '
                t=t+1
            items = line.strip().split(' ')    
            Y.append( np.eye(20)[np.array(items, dtype=np.int32)] )
            y_len.append(t)
            line = f.readline()
        f.close
        return Y,y_len
    def getdata(self):
        return         [self.emgtrainleft, self.emgtestleft , self.emgtrainright ,self.emgtestright ,
                        self.acctrainleft,self.acctestleft ,self.acctrainright,self.acctestright ,
        self.gyrtrainleft ,self.gyrtestleft ,self.gyrtrainright ,self.gyrtestright ,
        self.oltrainleft ,self.oltestleft ,self.oltrainright , self.oltestright ,
        self.oritrainleft , self.oritestleft ,self.oritrainright, self.oritestright ,
        self.ytrain,self.ytest,self.ohytr,self.ohyte,self.ohytr,self.tr_len,self.ohyte,self.te_len]

    def gettrainlabel(self):
        return self.y_trainlabel
    def gettestlabel(self):
        return self.y_testlabel    