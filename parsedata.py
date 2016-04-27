import json
import sys, os
from pprint import pprint

# The years we want to look at
years = ['2015']

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

# A dictionary with key NAME_ID_YEAR and values of dictionaries of stats
data = {}

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
            if tour['minRndsMet'] == 'Y': # OPTIONAL
                for cat in tour['statCats']:
                    # Get stats from all categories
                    for stat in cat['stats']:
                        name = stat['name']
                        sanitized[name] = {'rank': stat['rank'], 'value': stat['value']}
    
    if len(sanitized):
        data[player_name + '_' + player_number + '_' + player_year] = sanitized
    
    
pprint(data)

print('Retrieved %i players' % (len(data)))
