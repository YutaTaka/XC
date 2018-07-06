# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import os
import shutil

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Link,Chain,ChainList,optimizers,training,utils,Variable,serializers,cuda
import pickle

class MyModel(Chain):
    #パラメータ
    def __init__(self):
        super(MyModel,self).__init__(
            cn1=L.Convolution2D(3,20,5),
            cn2=L.Convolution2D(20,50,5),
            l1=L.Linear(186050,500),
            l2=L.Linear(500,5),
        )
        #ネットワークの接続、活性化関数
    def __call__(self,x):
        h1=F.max_pooling_2d(F.relu(self.cn1(x)),2)
        h2=F.max_pooling_2d(F.relu(self.cn2(h1)),2)
        h3=F.dropout(F.relu(self.l1(h2)))
        return self.l2(h3)


model = MyModel()
serializers.load_npz("../result/MyModel.npz",model)

# for a in range(len(test_cov)):
#     x=Variable(np.array([test_cov[a]],dtype=np.float32))
#     out=model(x)
#     ans=np.argmax(out.data)
#     print ans

files = os.listdir('../images/original256')

for file in files:
    if file == '.DS_Store':
        os.remove('../images/original256/.DS_Store')

files = os.listdir('../images/original256')

for file in files:
    pixdata = Image.open("../images/original256/" + file)
    pixdata = np.resize(pixdata,(256,256,3,1))
    pixdata = pixdata.transpose(3,2,0,1)
    x=Variable(np.array(pixdata,dtype=np.float32))
    out=model(x)
    ans=np.argmax(out.data)

    if ans == 0:
        shutil.copyfile("../images/original256/" + file, "../result/white/" + file)
    elif ans == 1:
        shutil.copyfile("../images/original256/" + file, "../result/yellow/" + file)
    elif ans == 2:
        shutil.copyfile("../images/original256/" + file, "../result/purple/" + file)
    elif ans == 3:
        shutil.copyfile("../images/original256/" + file, "../result/orange/" + file)
    elif ans == 4:
        shutil.copyfile("../images/original256/" + file, "../result/other/" + file)
