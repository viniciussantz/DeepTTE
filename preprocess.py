"""
Step-by-Step Methodology:

1. Temporal Split: 
   - Train: Everything before April 1, 2014.
   - Eval (Validation): April 2014 (approx. 30 days).
   - Test: May and June 2014 (approx. 60 days).

2. Spatial Resampling: Prevents the model from learning "trivial patterns" 
   by simply counting GPS records. The script forces the creation of a virtual point every 300 meters 
   (ignoring the original 15s rate). The model is forced to focus on the route's geometry.

3. Data leakage protection: Zeros out local gaps in test/eval sets to ensure 
   prediction relies solely on coordinates and global attributes.
"""

from __future__ import print_function, division
import csv
import json
import math
import os
import datetime

# Increase CSV limit to handle long Porto trajectories
csv.field_size_limit(10**8)

# Robust Temporal Split (Porto dataset ends June 30, 2014)
# TEST: 2 months (May and June)
# EVAL: 1 month (April)
TEST_DATA_START = datetime.datetime(2014, 5, 1)  
EVAL_DATA_START = datetime.datetime(2014, 4, 1)


def geo_distance(lon1, lat1, lon2, lat2):
    """Calculates Haversine distance in Kilometers."""
    lon1, lat1, lon2, lat2 = map(float, [lon1, lat1, lon2, lat2])
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return 6371.0 * 2 * math.asin(math.sqrt(a))

def resample_path(polyline, step_km=0.3):
    """
    Resamples a GPS trajectory to have points at fixed spatial intervals (e.g., 300 meters) 
    instead of fixed time intervals (15 seconds).
    
    How it works:
    It iterates through the original GPS points. If the distance between points is greater 
    than or equal to the target 'step_km', it uses linear interpolation to create a new 
    virtual point exactly at the target distance, calculating its precise longitude, 
    latitude, and estimated time.
    
    This prevents the neural network from "cheating" by simply counting the number 
    of GPS pings to guess the total travel time.
    """
    if len(polyline) < 2: return [], [], [], []
    
    new_lngs, new_lats = [polyline[0][0]], [polyline[0][1]]
    new_dist_gap, new_time_gap = [0.0], [0.0]
    
    accum_dist = 0.0
    current_dist_in_segment = 0.0
    
    for i in range(1, len(polyline)):
        d = geo_distance(polyline[i-1][0], polyline[i-1][1], polyline[i][0], polyline[i][1])
        while current_dist_in_segment + d >= step_km:
            ratio = (step_km - current_dist_in_segment) / d if d > 0 else 0
            res_lng = polyline[i-1][0] + ratio * (polyline[i][0] - polyline[i-1][0])
            res_lat = polyline[i-1][1] + ratio * (polyline[i][1] - polyline[i-1][1])
            res_time = ((i - 1) * 15.0) + (ratio * 15.0)
            
            new_lngs.append(res_lng)
            new_lats.append(res_lat)
            accum_dist += step_km
            new_dist_gap.append(accum_dist)
            new_time_gap.append(res_time)
            
            d -= (step_km - current_dist_in_segment)
            current_dist_in_segment = 0.0
            if d < 1e-7: break 
        current_dist_in_segment += d
    return new_lngs, new_lats, new_dist_gap, new_time_gap

def process_data(input_csv, output_dir, chunk_size=50000):
    """
    Main function to parse the dataset, apply temporal splitting, and format records.
    
    How it works:
    1. Reads the raw CSV row by row.
    2. Uses the timestamp to assign the trip to Train, Eval, or Test sets.
    3. Calls 'resample_path' to fix the spatial geometry.
    4. Applies Data leakage protection: Zeros out the local 'time_gap' and 'dist_gap' 
       for Eval and Test sets to prevent data leakage during model evaluation.
    5. Saves the processed records into separate chunk files when the buffer is full.
    """
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)

    stats = {'dist_gap': [], 'time_gap': [], 'lngs': [], 'lats': [], 'dist': [], 'time': []}
    taxi_id_map = {}
    taxi_id_counter = 0
    

    records = {'train': [], 'eval': [], 'test': []}
    chunk_idx = {'train': 0, 'eval': 0, 'test': 0}
    file_assignments = {'train': [], 'eval': [], 'test': []}

    
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            polyline = json.loads(row['POLYLINE'])
            if len(polyline) < 5: continue 

            ts = int(row['TIMESTAMP'])
            dt = datetime.datetime.utcfromtimestamp(ts)
            
            if dt >= TEST_DATA_START: split = 'test'
            elif dt >= EVAL_DATA_START: split = 'eval'
            else: split = 'train'

            lngs, lats, dist_gap, time_gap = resample_path(polyline, step_km=0.3)
            if len(lngs) < 5: continue 

            current_time_gap = time_gap if split == 'train' else [0.0] * len(lngs)
            current_dist_gap = dist_gap if split == 'train' else [0.0] * len(lngs)

            taxi_id = row['TAXI_ID']
            if taxi_id not in taxi_id_map:
                taxi_id_map[taxi_id] = taxi_id_counter
                taxi_id_counter += 1

            record = {
                'driverID': taxi_id_map[taxi_id],
                'dateID': dt.day,
                'weekID': dt.weekday(),
                'timeID': dt.hour * 60 + dt.minute,
                'dist': dist_gap[-1], 
                'time': ((len(polyline) - 1) * 15.0), 
                'lngs': lngs,
                'lats': lats,
                'states': [0] * len(lngs),
                'time_gap': current_time_gap,
                'dist_gap': current_dist_gap
            }
            
            records[split].append(record)

            if split == 'train':
                stats['lngs'].extend(lngs)
                stats['lats'].extend(lats)
                stats['dist'].append(record['dist'])
                stats['time'].append(record['time'])
                for j in range(2, len(dist_gap)):
                    stats['dist_gap'].append(dist_gap[j] - dist_gap[j-2])
                    stats['time_gap'].append(time_gap[j] - time_gap[j-2])

            if len(records[split]) >= chunk_size:
                fname = "{}_{:03d}".format(split, chunk_idx[split])
                save_chunk(records[split], output_dir, fname)
                file_assignments[split].append(fname)
                chunk_idx[split] += 1
                records[split] = []

    for split in ['train', 'eval', 'test']:
        if records[split]:
            fname = "{}_{:03d}".format(split, chunk_idx[split])
            save_chunk(records[split], output_dir, fname)
            file_assignments[split].append(fname)
    
    update_config(stats, file_assignments)

def save_chunk(records_list, output_dir, name):
    with open(os.path.join(output_dir, name), 'w') as f:
        for r in records_list: 
            f.write(json.dumps(r) + '\n')
    print("Saved file:", name)

def update_config(stats, assignments):
    def get_ms(arr):
        m = sum(arr)/len(arr) if arr else 0
        v = sum((x-m)**2 for x in arr)/len(arr) if arr else 1
        return m, math.sqrt(v)

    try:
        with open('config.json', 'r') as f: config = json.load(f)
    except FileNotFoundError:
        config = {} 

    print("\nCalculating normalization metrics...")
    for key in ['lngs', 'lats', 'dist', 'time', 'dist_gap', 'time_gap']:
        m, s = get_ms(stats[key])
        config[key + '_mean'], config[key + '_std'] = m, s
    
    config['train_set'] = assignments['train']
    config['eval_set'] = assignments['eval']
    config['test_set'] = assignments['test']
    
    with open('config.json', 'w') as f: 
        json.dump(config, f, indent=4)
        

if __name__ == '__main__':
    process_data('archive/train.csv', 'data')