#!/usr/bin/env python3
import configparser
import time
import datetime

import board
import busio

from adafruit_as726x import Adafruit_AS726x
from PushingBox import PushingBox

pbox = PushingBox()

startTime = time.time()
#maximum value for sensor reading
max_val = 16000

# Initialize I2C bus and sensor.
i2c = busio.I2C(board.SCL, board.SDA)
sensor = Adafruit_AS726x(i2c)

sensor.conversion_mode = sensor.MODE_2

# Initialize config details
config = configparser.ConfigParser()
config.read('config.ini')

# Use the Pi's MAC address as a unique identifier
mac = getMac()

# If Pi's MAC already exists in config, load the parameters
if mac in config:
    parameters = config[mac]
# If Pi's MAC is new, add to the config file and populate with default values
else:
    config[mac] = {}
    parameters = config['DEFAULT']

#Sampling frequency (seconds)
SAMPLING_FREQ = 60

#Device id for PushingBox
devid = parameters['pbox_devid']

def main():
    while True:
        # Wait for data to be ready
         while not sensor.data_ready:
             time.sleep(.1)

        print_values()
        write_data_to_local()

        if not all_dark():
            update_pushingbox()

        # Set sampling frequency
        while(time.time() - startTime <= SAMPLING_FREQ):
            time.sleep(1)
        startTime = startTime + SAMPLING_FREQ

#gets Mac Address of Pi as a string
def getMac(interface='eth0'):
    try:
        str = open ('/sys/class/net/net/%s/address', &interface).read()
    except:
        str = "00:00:00:00:00:00"
    return str[0:17]

#Returns bool regarding if light values exists (true) or not (false)
def all_dark():
        if sensor.violet != 0 or sensor.blue != 0 or sensor.green != 0 or
           sensor.yellow != 0 or sensor.orange != 0 or sensor.red != 0:
            return False
        return True

#prints values to console for debugging
def print_values():
    print('Violet: ' + str(sensor.violet))
    print('Blue: ' + str(sensor.blue))
    print('Green: ' + str(sensor.green))
    print('Yellow: ' + str(sensor.yellow))
    print('Orange: ' + str(sensor.orange))
    print('Red: ' + str(sensor.red))
    print("\n")

#attempts to write data to local reading file
def write_data_to_local():
    try:
        with open('as7262_readings.txt', 'a+') as write_file:
            write_file.write(str(datetime.datetime.now()) + "," + str(sensor.violet) +
                    "," + str(sensor.blue) + "," + str(sensor.green) + "," +
                    str(sensor.yellow) + "," + str(sensor.orange) + "," +
                    str(sensor.red) + "\n")
    except:
        with open('error_log.txt', 'a+') as error_file:
            error_file.write(str(datetime.datetime.now()) + " Could not write to file. Check remaining storage size.")

#updates pushing box with new sensor values.
def update_pushingbox():
    try:
        pbox.push(devid, violetData=sensor.violet, blueData=sensor.blue,
                greenData=sensor.green, yellowData=sensor.yellow,
                orangeData=sensor.orange, redData=sensor.red)
    except:
        with open('error_log.txt', 'a+') as error_file:
            error_file.write(str(datetime.datetime.now()) + " Could not push to PushingBox. Check internet connection.")


if __name__ == '__main__':
    main()
