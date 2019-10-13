
"""
stream.py

Contains classes to create different streamers for data from the HPWREN SDSC Davis weather station
"""

import os
import sys
import urllib.request
import numpy as np
import threading
import datetime
import time


class SDSCWeatherStreamer(object):
    """
    Description
    """
    def __init__(self, url_base, t_period=60):
        """
        Initialize EnvStreamer object with a url_base of the raw data from a Davis sensor with proper date formatting
        in the url to interface with datetime module
        """
        self._timer = None
        self.url_base = url_base
        self.url_prev = datetime.date.today().strftime(self.url_base)
        self.t_period = t_period
        self.t_next_read = time.time() + self.t_period
        self.t_streamed_up_to = time.time()
        self.b_running = False

        self.start_stream()

    def _execute(self):
        """
        Function triggered by the timer every self.t_period seconds
        """
        url_curr = datetime.date.today().strftime(self.url_base)
        print('{0:0.1f}: Accessing data from {1}'.format(time.time(), url_curr))

        # If day has rolled over, collect all information from previous day before continuing
        if (not(url_curr == self.url_prev)):
            with urllib.request.urlopen(url_curr) as fid:
                data_temp = fid.read()
                if (len(data_temp) > 0):
                    self.process_data(data_temp)

        with urllib.request.urlopen(url_curr) as fid:
            data_temp = fid.read()
            if (len(data_temp) > 0):
                self.process_data(data_temp)


        self.url_prev = url_curr

    def process_data(self, data_in):
        """
        Processes all data from time defined in self.t_streamed_up_to to current time, uploading all information
        to a Google Sheets doc
        """
        self.t_next_read = self.t_next_read + self.t_period
        self.b_running = False
        self.start_stream()

        for line in data_in.split(b'\n'):
            try:
                list_vars = line.split(b'\t')
                if (len(list_vars) == 4):
                    data_ip = list_vars[0]
                    data_name = list_vars[1]
                    data_timestamp = int(list_vars[2])
                    data_csv = list_vars[3].split(b',')
                else:
                    continue

                if (data_timestamp <= self.t_streamed_up_to):
                    continue

                print('  New data: {0}'.format(data_csv))

                # TODO Process the byte string into human-readable environmental variables

                # TODO Put in code to upload to Google Sheets

                # Update most recently read data
                self.t_streamed_up_to = data_timestamp

            except:
                print('ERROR: {0}'.format(line))
                print(sys.exc_info()[0])

    def start_stream(self):
        """
        Start the stream
        """
        if (not(self.b_running)):
            t_wait = self.t_next_read - time.time()
            if (t_wait < 0.1):
                t_wait = 1
            elif (t_wait > self.t_period):
                t_wait = self.t_period
            self._timer = threading.Timer(t_wait, self._execute)
            self._timer.start()
            self.b_running = True

    def stop_stream(self):
        """
        Stop the stream
        """
        self._timer.cancel()
        self.b_running = False


if __name__ == '__main__':
    t_period = 5
    url = 'http://hpwren.ucsd.edu/TM/Sensors/Data/%Y%m%d/198.202.124.3-HPWREN:SDSC:Davis:1:0'
    sdsc_weather_streamer = SDSCWeatherStreamer(url_base=url, t_period=t_period)

    try:
        while (True):
            time.sleep(1)

    finally:
        sdsc_weather_streamer.stop_stream()
