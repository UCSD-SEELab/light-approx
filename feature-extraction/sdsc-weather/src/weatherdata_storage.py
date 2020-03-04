import os
import sys
import requests
import urllib3 
import numpy as np
import threading
import datetime
import time
import json
from json import dumps
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials


 def process_data(data_in,filename):
       
        for line in data_in.split('\n'):
            #print('for loop in process_data')
            try:
                list_vars = line.split('\t')
                if (len(list_vars) == 4):
                    #print('if statement in try')
                    data_ip = list_vars[0]
                    data_name = list_vars[1]
                    data_timestamp = int(list_vars[2])
                    data_csv = list_vars[3].split(',')
                else:
                    continue

                #if (data_timestamp <= self.t_streamed_up_to):
                    
                    #continue

                print('  New data: {0}'.format(data_csv))

                # do numerical operation on all the elements
                Bar = int(data_csv[8]+data_csv[7],16)/29.83
                Bar = str(Bar)

                TempOut = int(data_csv[13]+data_csv[12],16)/10
                TempOut = float(TempOut)
                TempOut = str(((100.0/(212-32) * (TempOut - 32))*100)/100)

                TempIn = int(data_csv[10]+data_csv[9],16)/10
                TempIn = float(TempIn)
                TempIn = str(((100.0/(212-32) * (TempIn - 32))*100)/100)

                HumOut = str(int(data_csv[33],16))

                HumIn = str(int(data_csv[11],16))

                Wdir = str(int(data_csv[17]+data_csv[16],16))

                Wind = str(int(data_csv[14],16)/2.23694)

                Wind10 = str(int(data_csv[15],16)/2.23694)

                Solar = str(int(data_csv[45]+data_csv[44],16))

                UV = str(int(data_csv[43],16))

                RainRate = str(int(data_csv[42]+data_csv[41],16))

                DTime = datetime.datetime.fromtimestamp(data_timestamp).isoformat(' ')
                
                values = [DTime,Bar,TempIn,HumIn,TempOut,Wind,Wind10,Wdir,HumOut,RainRate,UV,Solar]
               
                with open(filename, 'a+') as f:
                    f.write("%s\n" % ', '.join(values))
                

            except Exception as e:
                print(e)
                print('ERROR: {0}'.format(line))
                print(sys.exc_info()[0])


def data_upload(scope,json,excel,txt):
    #scope = ["https://www.googleapis.com/auth/drive"]

    #creds = ServiceAccountCredentials.from_json_keyfile_name('test.json', scope)
    creds = ServiceAccountCredentials.from_json_keyfile_name(json, scope)
    client = gspread.authorize(creds)

    #sheet = client.open("test_sdsc").sheet1 # open sheet
    sheet = client.open(excel).sheet1
    data = pd.read_table(txt,delimiter=',',header=None)
    data = list(data.values.flatten())

    cell_list = sheet.range('A1:L7159') 
    i=0
    for cell in cell_list:
        cell.value = data[i]
        i+=1
    sheet.update_cells(cell_list)
    print('put data into Google Sheet')
    # Update most recently read data
                

def main():
    url = 'http://hpwren.ucsd.edu/TM/Sensors/Data/2019/20191201/198.202.124.3-HPWREN:SDSC:Davis:1:0'
    filename = '12_01_sdsc'
    with requests.get(url) as fid:
            data_temp = fid.text
            if (len(data_temp) > 0):
                process_data(data_temp,filename)
    
    scope = ["https://www.googleapis.com/auth/drive"]
    json = 'test.json'
    excel = 'test_sdsc'
    
    data_upload(scope,json,excel,filename)
