import sys, os, json
from pprint import pprint
import numpy as np

def player_data_from_years(years, require_min_rounds=False):
    # Find files matching a given year
    selected_files = []
    cwd = os.getcwd()
    # Iterate through player folders
    for player_folder in os.listdir(cwd + '/players'):
        # skip hidden folders
        if player_folder.startswith('.'): continue
        # get files in folder
        data_files = os.listdir(cwd + '/players/' + player_folder)
        # select files for given years
        for data_file in data_files:
            for year in years:
                if data_file.startswith(year) and data_file.endswith('stat.json'):
                    selected_files.append(cwd + '/players/' + player_folder + '/' + data_file)

    # An array of play stat dicts and values of dictionaries of stats
    data = []
    # Iterate through found files and parse
    for f in selected_files:
        # read json file
        file = open(f, 'r')
        json_data = json.load(file)
        file.close()

        # extract essentials from the json dictionary
        player_name = json_data['plrs'][0]['plrName']
        player_number = json_data['plrs'][0]['plrNum']
        player_year = json_data['plrs'][0]['years'][0]['year']

        # clean up dictionary
        sanitized = {}
        for tour in json_data['plrs'][0]['years'][0]['tours']:
            if tour['tourName'] == 'PGA TOUR':
                if require_min_rounds and tour['minRndsMet'] == 'N': continue
                # This is from the PGA TOUR
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

def extract_good_stats(feature_map, data):
    # Prune stats that don't show up in at least 80% of the entries
    num_records = len(data)
    requiredPercent = .8
    good_stats = {}
    for feature in feature_map:
        num_app = int(feature_map[feature]['appearances'])
        if num_app > requiredPercent * num_records:
            good_stats[feature] = num_app

    return good_stats

def build_stat_matrix(data, feature_map):
    """ Convert feature dictionaries to 2D arrays
    each row is a specific player and each column a specific stat """
    # get number of features
    n_features = len(feature_map)
    # gather features for each player
    player_stat_list = []
    for player in data:
        # initialize numpy array
        player_stats = np.empty((1, n_features))
        # iterate through stats and insert at appropriate index
        for stat in player['stats']:
            if stat not in feature_map: continue
            # get string value of given stat
            raw_val = player['stats'][stat]['value'].replace(',','')

            # parse to float, taking account of wonky formatting
            if '%' in raw_val: val = float(raw_val[:-1])/100
            elif '$' in raw_val: val = float(raw_val[1:])
            elif "'" in raw_val:
                split = raw_val.split("' ")
                val = int(split[0]) * 12 + int(split[1][:-1])
            else: val = float(raw_val)

            # add to array
            player_stats[0, feature_map[stat]] = val

        # append to list of stat arrays
        player_stat_list.append(player_stats)

    # concetenate list into single numpy array
    player_matrix = np.vstack(player_stat_list)
    return player_matrix

def extract_player_ranks(data):
    player_ranking_list = []
    # get each ranking
    for player in data:
        try: allaround = float(player['stats']['All-Around Ranking']['rank'])
        except KeyError: allaround = np.nan

        try: fedex = float(player['stats']['FedExCup Season Points']['rank'])
        except KeyError: fedex = np.nan

        try: money = float(player['stats']['Money Leaders']['rank'])
        except KeyError: money = np.nan

        player_ranking_list.append(np.array([allaround, fedex, money]))

    return np.vstack(player_ranking_list)

def main():
    # The years we want to look at
    years = ['2015', '2014', '2013']
    #years = [str(x) for x in range(1980, 2016)]

    # An array of dictionaries containing name, year, id, and stat dict
    data = player_data_from_years(years, True)

    # A dict from feature name to a dict containing {feature_index, num_appearances}
    feature_map = index_features_in_data(data)

    # Find which stats that actually appear in most of the players
    good_stats = extract_good_stats(feature_map, data)
    good_stats_keys = list(good_stats.keys())
    good_feature_map = {}
    for idx in range(len(good_stats)):
        good_feature_map[good_stats_keys[idx]] = idx

    # An array of dictionaries like in data that have all desired stats
    good_data = select_records_with_stats(data, good_stats, feature_map)

    # make matrix of the good data
    stat_matrix = build_stat_matrix(good_data, good_feature_map)
    rank_matrix = extract_player_ranks(good_data)

    # save things
    np.savetxt('stats.csv', stat_matrix, delimiter=',')
    np.savetxt('ranks.csv', rank_matrix, delimiter=',')

    print('Found %i unique features for %i players' % (len(feature_map), len(data)))
    print('Found %i records containing stats <%s>' % (len(good_data), ', '.join(good_stats)))

if __name__ == "__main__":
    main()
