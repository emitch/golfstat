import sys, os, json
from pprint import pprint
import numpy as np

# stat names corresponding to rankings, should be exluded in stat gathering
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
                    selected_files.append(
                        cwd + '/players/' + player_folder + '/' + data_file)

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
            data.append({'name': player_name, 'id': player_number, \
                'year': player_year, 'stats': sanitized})
            # data[player_name + '_' + player_number + '_' + player_year] = sanitized

    return data

def index_stats_in_data(reindex=False, required_fraction=0.5):
    """ create a single mapping from stats to integers that will
    be the same in all years.  To do this, run the indexing on
    the entire corpus and save the result to a file """

    if not reindex and os.path.isfile('stat_as_index.json'):
        print('Loading saved stat indexing...')
        with open('stat_as_index.json', 'r') as file:
            stat_as_index = json.load(file)
        with open('appearances.json', 'r') as file:
            appearances = json.load(file)
        index_as_stat = np.genfromtxt(
            'index_as_stat.csv', delimiter=',', dtype=str)
        entries_per_year = np.genfromtxt('entries_per_year.csv', delimiter=',')
    else:
        print('Indexing stats...')
        # load data from all years
        start_year = 1980
        end_year = 2015
        data = player_data_from_years(
            [str(x) for x in range(start_year, end_year+1)])

        # initialize empty data structures
        appearances = {}
        entries_per_year = np.zeros([end_year-start_year + 1])

        # keep track of the number of appearances of each stat
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
                if stat not in appearances:
                    appearances[stat] = 1
                else:
                    appearances[stat] += 1

        # prune stats that don't show up in the required fraction of records
        stat_count = 0
        stat_as_index = {}
        cutoff = len(data) * required_fraction
        # iterate through appearances and add passing stats to dictionary
        for stat in appearances:
            if appearances[stat] < cutoff: continue
            stat_as_index[stat] = stat_count
            stat_count += 1

        # do reverse mapping (index as stat)
        index_as_stat = [None]*len(stat_as_index)
        for stat in stat_as_index:
            index_as_stat[stat_as_index[stat]] = stat

        # save data
        with open('stat_as_index.json', 'w') as file:
            json.dump(stat_as_index, file)
        with open('appearances.json', 'w') as file:
            json.dump(appearances, file)
        with open('index_as_stat.csv', 'w') as file:
            for line in index_as_stat:
                file.write('%s,' % line)
        np.savetxt('entries_per_year.csv', entries_per_year, delimiter=',')

    return stat_as_index, index_as_stat, appearances, entries_per_year

# Select only those records that have the given stats (name or index)
def select_records_with_stats(records, stats):
    passing_records = []
    for player in records:
        player_data = player['stats']
        has_all_stats = True
        for stat in stats:
            stat_name = stat
            if stat_name not in player_data:
                has_all_stats = False
                break

        if has_all_stats:
            passing_records.append(player)

    return passing_records

def build_stat_matrix(data, stat_as_index):
    """ Convert stat dictionaries to 2D arrays
    each row is a specific player and each column a specific stat """
    # get number of stats
    n_stats = len(stat_as_index)
    n_players = len(data)

    # initialize numpy array to NaNs
    player_matrix = np.empty([n_players, n_stats], dtype=float)
    player_matrix.fill(np.nan)
    for i, player in enumerate(data):
        # iterate through stats and insert at appropriate index
        for stat in player['stats']:
            if stat not in stat_as_index: continue
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
            player_matrix[i, stat_as_index[stat]] = val

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

def gather(years, stat_as_index=None, index_as_stat=None):
    # build dictionaries
    if stat_as_index is None or index_as_stat is None:
        stat_as_index, index_as_stat, _, _ = index_stats_in_data()
    data = player_data_from_years(years)

    # convert to matrix and get ranks
    ranks = extract_player_ranks(data)
    # names = extract_player_names(data)
    stats = build_stat_matrix(data, stat_as_index)

    return stats, ranks, index_as_stat, stat_as_index

if __name__ == "__main__":
    # Re-index stats
    index_stats_in_data(reindex=True)

    # The years we want to look at
    years = ['2013', '2014', '2015']
    stat_matrix, rank_matrix, index_as_stat, stat_names = gather(years)

    print('Found %i records with %i unique stats' % stat_matrix.shape)
