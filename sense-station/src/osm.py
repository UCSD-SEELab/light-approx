import requests
import time
import pandas as pd

def send_request(curr_lat, curr_long):
    params = (
            ('format', 'jsonv2'),
            ('lat', curr_lat),
            ('lon', curr_long),
            ('zoom', '18'),
            ('addressdetails', '1'),
            )

    r = requests.get('https://nominatim.openstreetmap.org/reverse', params=params)
    return r

def check_indoor(location_type):
    indoor_categories = ['building', 'amenity', 'shop']
    if location_type in indoor_categories:
        return 1
    else:
        return 0

if __name__ == '__main__':
    read_file = input("Enter the absolute path to input csv file: ")
    data = pd.read_csv(read_file)

    # Assign indices to each datapoint
    idx = list(map(str,range(0,data['latitude'].size)))
    data['idx'] = idx

    io = []
    isIndoor = []

    last_lat = None
    last_long = None
    last_location = None

    for i in range(0, data['latitude'].size):
        curr_lat = data['latitude'][i]
        curr_long = data['longitude'][i]
        if curr_lat == last_lat and curr_long == last_long:
            location_type = last_location
        else:
            r = send_request(curr_lat, curr_long)
            location_type = r.json()['category']
            time.sleep(1.5) # Prevent overloading requests to API

        io.append(location_type)

        isIndoor.append(check_indoor(location_type))

        last_lat = curr_lat
        last_long = curr_long
        last_location = location_type

        # print("{} : {}".format(data['idx'][i], io[i]))

    data['location_type'] = io
    data['isIndoor'] = isIndoor
    data.to_csv(read_file, index=False)

