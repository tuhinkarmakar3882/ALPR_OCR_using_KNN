import json  
import pandas as pd  
from pandas.io.json import json_normalize
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import cv2
from progressbar import *


widgets = ['>>> Total Progress: ', Percentage(), ' ', Bar(marker='#',left='[',right=']'),
           ' ', ETA(), ' ', FileTransferSpeed()] #see docs for other options




def save_img(img,num):
    cv2.imwrite("training_images/"+str(num)+".jpg",img)
    

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image,1



#===========< MAIN >=====================================================================================


print(">>> Opening Json File")
with open('INP3.json') as my_json_file: 
    data_file = json.load(my_json_file)


print(">>> Normalizing Json File")
content_urls = json_normalize(data_file['dataset_elements'])

print("Looking for last Download...")
last_pos_file = open('last_pos.txt','r')
for value in last_pos_file:
	last_pos = int(value)
last_pos_file.close()
print("Last Download Possition = " + str(last_pos) + "\n")

print(">>> Starting to Download the Images...")

current_image_num = last_pos
all_images=[]


pbar = ProgressBar(widgets=widgets, maxval=len(content_urls)+1)
seek = last_pos
pbar.start()
a=0
for link in range(last_pos,len(content_urls.content)):
	flag = False
	while not flag:
		try:
			current_image , flag = url_to_image(content_urls.content[link])
		except:
			a+=1
	save_img(current_image, current_image_num)
	current_image_num+=1
	seek+=1
	last_pos_file = open('last_pos.txt','w')
	last_pos_file.write(str(last_pos))
	last_pos_file.close()
	last_pos+=1
	pbar.update(seek)
pbar.finish()

print("All files are successfully saved")
