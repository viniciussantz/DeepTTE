from __future__ import print_function, division
import csv
import json
import math
import os
import random
import datetime
from geopy.distance import geodesic

# Haversine distance
def geo_distance(lon1, lat1, lon2, lat2):
    return geodesic((float(lat1), float(lon1)), (float(lat2), float(lon2))).km

def process_data(input_csv, output_dir, chunk_size=50000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Statistics accumulators
    stats = {
        'dist_gap': [], # We will collect 2-step increments (assuming kernel_size=3)
        'time_gap': [], # 1-step increment is always 15, 2-step is 30. We'll store 2-step.
        'lngs': [],
        'lats': [],
        'dist': [],
        'time': []
    }

    taxi_id_map = {}
    taxi_id_counter = 0

    chunk_idx = 0
    records = []

    print("Processing CSV...")
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if row['MISSING_DATA'] == 'True':
                continue
            
            polyline = json.loads(row['POLYLINE'])
            if len(polyline) < 10:
                continue

            lngs = [p[0] for p in polyline]
            lats = [p[1] for p in polyline]

            # Cumulative distances and times
            dist_gap = [0.0]
            time_gap = [0.0]
            for j in range(1, len(polyline)):
                d = geo_distance(lngs[j-1], lats[j-1], lngs[j], lats[j])
                dist_gap.append(dist_gap[-1] + d)
                time_gap.append(time_gap[-1] + 15.0)

            total_dist = dist_gap[-1]
            total_time = time_gap[-1]

            # Taxonomy mappings
            taxi_id = row['TAXI_ID']
            if taxi_id not in taxi_id_map:
                taxi_id_map[taxi_id] = taxi_id_counter
                taxi_id_counter += 1
            driverID = taxi_id_map[taxi_id]

            timestamp = int(row['TIMESTAMP'])
            dt = datetime.datetime.utcfromtimestamp(timestamp)
            timeID = dt.hour * 60 + dt.minute
            weekID = dt.weekday()
            dateID = dt.timetuple().tm_yday # 1 to 365/366

            states = [0.0] * len(polyline)

            # Record
            record = {
                'driverID': driverID,
                'dateID': dateID,
                'weekID': weekID,
                'timeID': timeID,
                'dist': total_dist,
                'time': total_time,
                'lngs': lngs,
                'lats': lats,
                'states': states,
                'time_gap': time_gap,
                'dist_gap': dist_gap
            }
            records.append(record)

            # Only collect stats up to 200,000 to keep memory low
            if i <= 200000:
                for j in range(2, len(dist_gap)):
                    stats['dist_gap'].append(dist_gap[j] - dist_gap[j-2])
                
                stats['lngs'].extend(lngs)
                stats['lats'].extend(lats)
                stats['dist'].append(total_dist)
                stats['time'].append(total_time)

            if len(records) >= chunk_size:
                out_name = "train_{:02d}".format(chunk_idx)
                with open(os.path.join(output_dir, out_name), 'w') as out_f:
                    for r in records:
                        out_f.write(json.dumps(r) + '\n')
                print("Saved", out_name)
                chunk_idx += 1
                records = []

    if records:
        out_name = "train_{:02d}".format(chunk_idx)
        with open(os.path.join(output_dir, out_name), 'w') as out_f:
            for r in records:
                out_f.write(json.dumps(r) + '\n')
        print("Saved", out_name)

    print("Calculating statistics...")
    def calc_mean_std(arr):
        if not arr: return 0.0, 1.0
        mean = sum(arr) / len(arr)
        variance = sum((x - mean) ** 2 for x in arr) / len(arr)
        std = math.sqrt(variance)
        return mean, std if std > 1e-5 else 1.0

    config_update = {}
    for key in ['dist_gap', 'lngs', 'lats', 'dist', 'time']:
        m, s = calc_mean_std(stats[key])
        config_update[key + '_mean'] = m
        config_update[key + '_std'] = s
    
    # For time_gap we know it's 15s per step.
    # The model DeepTTE multiplies config['time_gap_mean'] by (kernel_size - 1).
    # This means config['time_gap_mean'] should be the 1-step increment!
    config_update['time_gap_mean'] = 15.0
    config_update['time_gap_std'] = 1.0 # arbitrary non-zero
    
    # Let's fix dist_gap_mean to be 1-step increment to see if it matches.
    # Actually GeoConv doesn't multiply by k-1. We'll stick to our 2-step mean and just use it.
    
    print("New config values:")
    print(json.dumps(config_update, indent=4))
    
    # Re-writing the config.json
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    config.update(config_update)
    
    # Update sets
    all_files = ["train_{:02d}".format(i) for i in range(chunk_idx+1)]
    # Split: last 1 for eval, last 1 for test, rest for train
    if len(all_files) >= 3:
        config['train_set'] = all_files[:-2]
        config['eval_set'] = [all_files[-2]]
        config['test_set'] = [all_files[-1]]
    else:
        config['train_set'] = all_files
        config['eval_set'] = all_files
        config['test_set'] = all_files

    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    print("Preprocessing completed and config.json updated.")

if __name__ == '__main__':
    process_data('archive/train.csv', 'data')
