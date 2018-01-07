# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 17:59:55 2017

@author: Tim
"""

import os

x = 0
i = 0
path = r'C:/Users/Tim/pythonscripts/soil/'
files = os.listdir(path)


for i, file in enumerate(files):
    os.rename((os.path.join(path, file)), os.path.join(path + 'soil' + str(i) + '.jpg'))
    