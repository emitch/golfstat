import sys, os, json
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pylab as pl
import imageio
import operator

# stat names corresponding to rankings, should be exluded in stat gathering
rank_stats = ['All-Around Ranking', 'FedExCup Season Points', 'Money Leaders']

def stat_files_for_years(years):
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

    return selected_files

def tournament_files_for_years(years):
    selected_files = []
    cwd = os.getcwd()
    tournaments_by_year = {}

    # Iterate through tournaments
    for tournament_folder in os.listdir(cwd + '/scorecards'):
        # skip hidden folders
        if tournament_folder.startswith('.'): continue

        # get years for this tournament
        year_folders = os.listdir(cwd + '/scorecards/' + tournament_folder)

        tournaments_by_year[tournament_folder] = {}

        # select files for given years
        for year_folder in year_folders:
            # skip years we don't need
            if year_folder not in years: continue

            tournaments_by_year[tournament_folder][year_folder] = []

            # skip hidden folders
            if tournament_folder.startswith('.'): continue

            player_files = os.listdir(cwd + '/scorecards/' + tournament_folder + '/' + year_folder)

            # finally we have all of the players
            for player_file in player_files:
                tournaments_by_year[tournament_folder][year_folder].append({'f': cwd + '/scorecards/' + tournament_folder + '/' + year_folder + '/' + player_file, 'id': player_file.split('.')[0]})

    return tournaments_by_year

def player_data_from_years(years, require_min_rounds=True, dict_by_id=False):
    # Find files matching a given year
    selected_files = stat_files_for_years(years)

    # An array of play stat dicts and values of dictionaries of stats
    if dict_by_id:
        data = {}
    else:
        data = []

    # Iterate through found files and parse
    for f in selected_files:
        # read json file
        file = open(f, 'r')
        json_data = json.load(file)
        file.close()

        # extract essentials from the json dictionary
        try:
            player_name = json_data['plrs'][0]['plrName']
            player_number = json_data['plrs'][0]['plrNum']
            player_year = json_data['plrs'][0]['years'][0]['year']
        except:
            sys.stdout.write('\r')
            print(f)
            continue

        sys.stdout.write('\r%s\t%s' % (player_number, player_year))
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
            if dict_by_id:
                if player_number not in data:
                    data[player_number] = {'name': player_name, 'years': {}}

                if player_year not in data[player_number]:
                    data[player_number][player_year] = {}

                data[player_number][player_year]['stats'] = sanitized
            else:
                data.append({'name': player_name, 'id': player_number, \
                    'year': player_year, 'stats': sanitized})
            # data[player_name + '_' + player_number + '_' + player_year] = sanitized

    print('')

    # NOW WE HAVE A DICT OF STATS
    # READ IN TOURNAMENT DATA
    tournament_file_dict = tournament_files_for_years(years)
    leaderboard_dict = {}

    for tournament in tournament_file_dict:
        # Get the course data file
        print('=====================')
        print('Tournament %s' % (tournament))
        
        leaderboard_dict[tournament] = {}
        
        for year in tournament_file_dict[tournament]:
            course_file_path = '/'.join(tournament_file_dict[tournament][year][0]['f'].split('/')[:-1]) + '/course.json'
            course_file = open(course_file_path, 'r')
            course_data = json.load(course_file)
            course_file.close()
            course_name = '>> NO COURSE NAME <<'
            if len(course_data['courses']):
                course_name = course_data['courses'][0]['name']
            print('Got data from %s (%s)' % (course_name, year))
            
            leaderboard_dict[tournament][year] = []
            
            for idx, player in enumerate(tournament_file_dict[tournament][year]):
                pid = player['id']
                if pid in data and year in data[pid]:
                    file = open(tournament_file_dict[tournament][year][idx]['f'], 'r')
                    scorecard_data = json.load(file)
                    file.close()

                    player = tournament_file_dict[tournament][year][idx]['id']

                    this_player_tournament_data = {}

                    # print('Player %s' % (pid))

                    this_player_tournament_data['scorecard'] = scorecard_data
                    summary = {}

                    total_rounds = len(scorecard_data['p']['rnds'])

                    # print('\t%s rounds' % (total_rounds))

                    hole_scores = []

                    valid_data = True

                    for round in scorecard_data['p']['rnds']:
                        for hole in round['holes']:
                            if len(hole['sc']) == 0:
                                valid_data = False
                                break

                            hole_scores.append(int(hole['sc']))

                    # print('\tTotal Shots: %i' % (sum(hole_scores)))

                    if valid_data == False:
                        continue

                    if total_rounds != 2 and total_rounds != 4 and total_rounds != 3:
                        continue
                        
                    summary['num_rounds'] =     total_rounds
                    summary['total_shots'] =    sum(hole_scores)
                    summary['course_name'] =    course_name
                    
                    this_player_tournament_data['summary'] = summary

                    data[pid][year][tournament] = this_player_tournament_data
                    
                    leaderboard_dict[tournament][year].append({'id': player, 'score': sum(hole_scores), 'mc': total_rounds != 4})

    for tournament in leaderboard_dict:
        for year in leaderboard_dict[tournament]:
            leaderboard = sorted(leaderboard_dict[tournament][year], key=operator.itemgetter('score'))
            rank = 0
            best_score = 0
            leaderboard_start = -1
            for idx, player in enumerate(leaderboard):
                if player['mc']:
                    data[player['id']][year][tournament]['summary']['rank'] = 'mc'
                else:
                    if leaderboard_start == -1:
                        leaderboard_start = idx
                        
                    if player['score'] > best_score:
                        best_score = player['score']
                        rank = idx + 1 - leaderboard_start
                    data[player['id']][year][tournament]['summary']['rank'] = rank
                    player['rank'] = rank
            
            leaderboard_dict[tournament][year] = leaderboard
                    

    # pprint(leaderboard_dict)
    pprint(data)

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

def show_stat_over_time(data, stat_name, years):
    stat_distributions_by_year = {}
    for year in years:
        stat_distributions_by_year[year] = {}

    min_val = sys.maxint
    max_val = -sys.maxint
    for player_data in data:
        stats = player_data['stats']
        for stat in stats:
            year = player_data['year']
            if year in years:
                if stat not in stat_distributions_by_year[year]:
                    stat_distributions_by_year[year][stat] = []
                stat_distributions_by_year[year][stat].append(stats[stat]['value'])
                if stat == stat_name:
                    value = stats[stat]['value'].replace('%', '').replace('$', '').replace(',', '')
                    value = float(value)

                    if value < min_val: min_val = value
                    if value > max_val: max_val = value

    print('\nFound:')
    for stat in stat_distributions_by_year[years[0]]: print(stat)
    print('---------------\nBuilding GIF for', stat_name)

    image_files = []

    writer = imageio.get_writer(os.getcwd() + '/' + stat_name + '.gif', fps=5)

    for idx, year in enumerate(years):
        if stat_name in stat_distributions_by_year[year]:
            dist = [float(s.replace('%', '').replace('$', '').replace(',', '')) for s in stat_distributions_by_year[year][stat_name]]
            plt.figure()
            plt.hist(dist, bins=30, range=[min, max])
            plt.title(year + ' ' + stat_name + ' ' + "{0:.2f}".format(sum(dist)/len(dist)))
            name = str(idx) + '.png'
            plt.savefig(name, bbox_inches='tight')
            writer.append_data(imageio.imread(os.getcwd() + '/' + name))
            os.remove(os.getcwd() + '/' + name)

    writer.close()

    print('Successfully built GIF for', stat_name)

if __name__ == "__main__":
    # Re-index stats
    # index_stats_in_data(reindex=True)

    # The years we want to look at
    years = [str(y) for y in range(2013, 2017)]
    
    data = player_data_from_years(years, dict_by_id=True)
    # pprint(data)

    # show_stat_over_time(data, 'Putts Per Round', years)

    # stat_matrix, rank_matrix, index_as_stat, stat_names = gather(years)

    # print('Found %i records with %i unique stats' % stat_matrix.shape)
