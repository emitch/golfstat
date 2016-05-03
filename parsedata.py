import sys, os, json
from pprint import pprint
import numpy as np

# stat names corresponding to rankings, should be exluded in feature gathering
rank_stats = ['All-Around Ranking', 'FedExCup Season Points', 'Money Leaders']

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

def index_features_in_data(reindex=False):
    """ create a single mapping from features to integers that will
    be the same in all years.  To do this, run the indexing on
    the entire corpus and save the result to a file """

    if not reindex and os.path.isfile('feature_as_index.json'):
        print('Loading saved feature indexing...')
        with open('feature_as_index.json', 'r') as file:
            feature_as_index = json.load(file)
        with open('feature_appearances.json', 'r') as file:
            feature_appearances = json.load(file)
        index_as_feature = np.genfromtxt(
            'index_as_feature.csv', delimiter=',', dtype=str)
    else:
        print('Indexing features...')
        # load data from all years
        start_year = 1980
        end_year = 2015
        data = player_data_from_years(
            [str(x) for x in range(start_year, end_year+1)])

        # initialize empty data structures
        feature_as_index = {}
        feature_appearances = {}
        entries_per_year = np.zeros([end_year-start_year + 1])
        feature_count = 0

        # iterate through players and add each appearing stat to the map
        # also keep track of the number of appearances of each stat
        for player in data:
            # get year data
            year = int(player['year'])
            entries_per_year[year-start_year] += 1
            # iterate through stats
            record = player['stats']
            for stat in record:
                # exclude stats corresponding to rankings
                if stat in rank_stats: continue
                # add unseen stats to dictionary
                if stat not in feature_as_index:
                    feature_as_index[stat] = feature_count
                    feature_appearances[stat] = 1
                    feature_count += 1
                # otherwise update number of appearances
                else:
                    feature_appearances[stat] += 1

        # do reverse mapping (index as feature)
        index_as_feature = [None]*len(feature_as_index)
        for feature in feature_as_index:
            index_as_feature[feature_as_index[feature]] = feature

        # save data
        with open('feature_as_index.json', 'w') as file:
            json.dump(feature_as_index, file)
        with open('feature_appearances.json', 'w') as file:
            json.dump(feature_appearances, file)
        with open('index_as_feature.csv', 'w') as file:
            for line in index_as_feature:
                file.write('%s,' % line)

    return feature_as_index, index_as_feature, feature_appearances

# Select only those records that have the given stats (name or index)
def select_records_with_stats(records, stats):
    passing_records = []
    for player in records:
        player_data = player['stats']
        has_all_stats = True
        for stat in stats:
            feature_name = stat
            if feature_name not in player_data:
                has_all_stats = False
                break

        if has_all_stats:
            passing_records.append(player)

    return passing_records

"""
# TODO: combine this with index_features_in_data
# THIS IS BROKEN, don't use with other functions
def extract_good_stats(feature_appearances, feature_as_index, data, exclude=None):
    # Prune stats that don't show up in at least 80% of the entries
    num_records = len(data)
    required_percent = .8
    good_stats = {}
    for feature in feature_map:
        if feature in exclude: continue
        num_app = feature_appearances[feature]
        if num_app > required_percent * num_records:
            good_stats[feature] = feature_as_index[feature]

    # # re-index
    # for stat, idx in zip(good_stats, range(len(good_stats))):
    #     good_stats[stat] = idx

    # dictionary mapping stats to indices
    return good_stats
"""

def build_stat_matrix(data, feature_as_index):
    """ Convert feature dictionaries to 2D arrays
    each row is a specific player and each column a specific stat """
    # get number of features
    n_features = len(feature_as_index)
    n_players = len(data)

    # initialize numpy array to NaNs
    player_matrix = np.empty([n_players, n_features], dtype=float)
    player_matrix.fill(np.nan)
    for i, player in enumerate(data):
        # iterate through stats and insert at appropriate index
        for stat in player['stats']:
            if stat not in feature_as_index: continue
            # get string value of given stat
            raw_val = player['stats'][stat]['value'].replace(',','')

            # parse to float, taking account of wonky formatting
            if '%' in raw_val: val = float(raw_val[:-1])/100
            elif '$' in raw_val: val = float(raw_val[1:])
            elif "'" in raw_val:
                # convert distance to inches
                split = raw_val.split("' ")
                val = int(split[0]) * 12 + int(split[1][:-1])
            else:
                try: val = float(raw_val)
                except: continue

            # add to array
            player_matrix[i, feature_as_index[stat]] = val

    return player_matrix

def extract_player_ranks(data):
    """ Extract the three rankings used by PGA from the player stat dictionaries"""
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

def extract_player_names(data):
    player_names = []
    for player in data:
        player_names.append(player['name'])

    return player_names

def gather(years, feature_as_index=None, index_as_feature=None):
    # build dictionaries
    if feature_as_index is None or index_as_feature is None:
        feature_as_index, index_as_feature, _ = index_features_in_data()
    data = player_data_from_years(years)
    # # clean ups
    # exclude = ['All-Around Ranking', 'FedExCup Season Points', 'Money Leaders']
    # good_stats = extract_good_stats(feature_map, data, exclude)
    # data = select_records_with_stats(data, good_stats, feature_map)

    # convert to matrix and get ranks
    ranks = extract_player_ranks(data)
    # names = extract_player_names(data)
    stats = build_stat_matrix(data, feature_as_index)

    return stats, ranks, index_as_feature, feature_as_index

if __name__ == "__main__":
    # The years we want to look at
    years = ['2013', '2014', '2015']
    stat_matrix, rank_matrix, index_as_feature, stat_names = gather(years)

    print('Found %i records with %i unique features' % stat_matrix.shape)
