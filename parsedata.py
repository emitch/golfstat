import json
import sys, os
from pprint import pprint

def player_data_from_years(years):
    selected_files = []
    cwd = os.getcwd()

    for player_folder in os.listdir(cwd + '/players'):
        if player_folder.startswith('.'): continue
        
        data_files = os.listdir(cwd + '/players/' + player_folder)
        year_matches = []
        for data_file in data_files:
            for year in years:
                if data_file.startswith(year) and data_file.endswith('stat.json'):
                    selected_files.append(cwd + '/players/' + player_folder + '/' + data_file)

    # An array of play stat dicts and values of dictionaries of stats
    data = []

    for f in selected_files:
        file = open(f, 'r')
        json_data = json.load(file)
        file.close()
        
        player_name = json_data['plrs'][0]['plrName']
        player_number = json_data['plrs'][0]['plrNum']
        player_year = json_data['plrs'][0]['years'][0]['year']
        
        sanitized = {}
        
        for tour in json_data['plrs'][0]['years'][0]['tours']:
            if tour['tourName'] == 'PGA TOUR':
                # This is from the PGA TOUR
                # if tour['minRndsMet'] == 'Y': # OPTIONAL
                for cat in tour['statCats']:
                    # Get stats from all categories
                    for stat in cat['stats']:
                        name = stat['name']
                        # Normalize names for stat names that have changed
                        name = name.replace(' - ', ': ')
                        sanitized[name] = {'rank': stat['rank'], 'value': stat['value']}
        
        if len(sanitized):
            data.append({'name': player_name, 'id': player_number, 'year': player_year, 'stats': sanitized})
            # data[player_name + '_' + player_number + '_' + player_year] = sanitized
    
    return data

def index_features_in_data(data):
    feature_index_map = {}
    feature_count = 0
    for player in data:
        record = player['stats']
        for stat in record:
            if stat not in feature_index_map:
                feature_index_map[stat] = {'feature_index': feature_count, 'appearances': 1}
                feature_count += 1
            else:
                feature_index_map[stat]['appearances'] += 1
    
    return feature_index_map
    
# Select only those records that have the given stats (name or index)
def select_records_with_stats(records, stats, feature_map):
    passing_records = []
    for player in records:
        player_data = player['stats']
        has_all_stats = True
        for stat in stats:
            feature_name = ""
            if isinstance(stat, int):
                # THIS DOESN'T WORK YET
                feature_name = list(feature_map.keys())[list(feature_map.values()).index(stat)]
            else:
                feature_name = stat
            if feature_name not in player_data:
                has_all_stats = False
                break
        
        if has_all_stats:
            passing_records.append(player)

    return passing_records

# The years we want to look at
years = ['2015', '2014', '2013']
stats = ["Putting Average", "Total Driving", "Putting from 6'", "Scrambling", "Ball Striking", "Strokes Gained: Putting"]

# An array of dictionaries containing name, year, id, and stat dict
data = player_data_from_years(years)

# A dict from feature name to a dict containing {feature_index, num_appearances}
feature_map = index_features_in_data(data)

# An array of dictionaries like in data that have all desired stats
passing = select_records_with_stats(data, stats, feature_map)

# print(json.dumps(feature_map, indent=4))
print('Found %i unique features for %i players' % (len(feature_map), len(data)))
print('Found %i records containing stats <%s>' % (len(passing), ', '.join(stats)))
