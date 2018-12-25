# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 11:36:58 2018

@author: 1mmm
"""
import numpy as np
import copy
class nlp(object):
    def __init__(self, x):
        self.v=[1,2,3,4,5,6,7,9,10,11,12]
        
        self.adv=[13]
        
        self.jie=[8,14]
        
        self.n=[15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
        
        self.dai=[30]
        
        self.maxn=[]
        
        self.maxa=[]
        
        self.maxp=[]
        
        self.ans=[0]


        self.parse(x)
   
    def parse(self,x,num=0):
        x1=copy.deepcopy(x)
        for i in range(1,36):
            if not (i in self.n):
                x1[num][i]=0
        self.step1(x1,num+1)
        
        x1=copy.deepcopy(x)
        for i in range(1,36):
            if not (i in self.adv):
                x1[num][i]=0
        self.step2(x1,num+1)
        
        x1=copy.deepcopy(x)
        for i in range(1,36):
            if not (i in self.jie):
                x1[num][i]=0
        self.step3(x1,num+1)
        
        x1=copy.deepcopy(x)
        for i in range(1,36):
            if not (i in self.v):
                x1[num][i]=0
        self.step4(x1,num+1)

        x1=copy.deepcopy(x)
        for i in range(1,36):
            if not (i in self.dai):
                x1[num][i]=0
        self.step8(x1,num+1)        
        mn=0
        mp=0
        mx=[]
        for i in range(len(self.maxp)):
            if self.maxn[i]>mn:
                mp=self.maxp[i]
                mn=self.maxn[i]
                mx=self.maxa[i]
            else:
                if self.maxn[i]==mn:
                    if self.maxp[i]>mp:
                        mp=self.maxp[i]
                        mn=self.maxn[i]
                        mx=self.maxa[i]
        self.ans=mx
    
                            
    def step1(self,x,num):
        x1=copy.deepcopy(x)
        for i in range(1,36):
            if not (i in self.v):
                x1[num][i]=0
        self.step4(x1,num+1)
        
        x1=copy.deepcopy(x)
        for i in range(1,36):
            if not (i in self.jie):
                x1[num][i]=0
        self.step3(x1,num+1)
    
        x1=copy.deepcopy(x)
        for i in range(1,36):
            if not (i in self.adv):
                x1[num][i]=0
        self.step2(x1,num+1)
    
    def step2(self,x,num):
        
        x1=copy.deepcopy(x)
        for i in range(1,36):
            if not (i in self.v):
                x1[num][i]=0
        self.step4(x1,num+1)    
        
    def step3(self,x,num):

        x1=copy.deepcopy(x)
        for i in range(1,36):
            if not (i in self.adv):
                x1[num][i]=0
        self.step2(x1,num+1)

        x1=copy.deepcopy(x)
        for i in range(1,36):
            if not (i in self.v):
                x1[num][i]=0
        self.step4(x1,num+1)    
        
    
        x1=copy.deepcopy(x)
        for i in range(1,36):
            if not (i in self.n):
                x1[num][i]=0
        self.step6(x1,num+1)   
    
    def step4(self,x,num):
        a=np.argmax(x,1)

        pc=1
        ans=[]
        f=True
        for i in range(num):
            if x[i][a[i]]>0.1:
                ans.append(a[i])
                pc=pc*x[i][a[i]]
            else:
                f=False
        if f:
            self.maxp.append(pc)
            self.maxa.append(ans)
            self.maxn.append(num)   
        if num<3:
            x1=copy.deepcopy(x)
            for i in range(1,36):
                if not (i in self.dai):
                    x1[num][i]=0
            self.step8(x1,num+1)
            x1 = copy.deepcopy(x)
            for i in range(1, 36):
                if not (i in self.v):
                    x1[num][i] = 0
            self.step9(x1, num + 1)
        if num<4:
            x1=copy.deepcopy(x)
            for i in range(1,36):
                if not (i in self.n):
                    x1[num][i]=0
            self.step5(x1,num+1)   
        
    def step5(self,x,num):
        a=np.argmax(x,1)
        pc=1
        ans=[]
        f=True
        for i in range(num):
            if x[i][a[i]]>0.1:
                ans.append(a[i])
                pc=pc*x[i][a[i]]
            else:
                f=False
        if f:
            self.maxp.append(pc)
            self.maxa.append(ans)
            self.maxn.append(num) 
        
    def step6(self,x,num):
        a=np.argmax(x,1)
        pc=1
        ans=[]
        f=True
        for i in range(num):
            if x[i][a[i]]>0.1:
                ans.append(a[i])
                pc=pc*x[i][a[i]]
            else:
                f=False
        if f:
            self.maxp.append(pc)
            self.maxa.append(ans)
            self.maxn.append(num)     
        x1=copy.deepcopy(x)
        for i in range(1,36):
            if not (i in self.v):
                x1[num][i]=0
        self.step7(x1,num+1)    
    
    def step7(self,x,num):
        a=np.argmax(x,1)
        pc=1
        ans=[]
        f=True
        for i in range(num):
            if x[i][a[i]]>0.1:
                ans.append(a[i])
                pc=pc*x[i][a[i]]
            else:
                f=False
        if f:
            self.maxp.append(pc)
            self.maxa.append(ans)
            self.maxn.append(num) 
    def step8(self,x,num):
        
        x1=copy.deepcopy(x)
        for i in range(1,36):
            if not (i in self.n):
                x1[num][i]=0
        self.step5(x1,num+1)

    def step9(self, x, num):

        x1 = copy.deepcopy(x)
        for i in range(1, 36):
            if not (i in self.n):
                x1[num][i] = 0
        self.step5(x1, num + 1)
    def getans(self):
        if len(self.ans)==0:
            self.ans=[0]
        return self.ans
    
    def getas(self):
        return self.maxa
        