import sys, os, json, re
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pylab as pl
import imageio
import operator

# stat names corresponding to rankings, should be exluded in stat gathering
rank_stats = ['All-Around Ranking', 'FedExCup Season Points',
'Money Leaders', 'Official World Golf Ranking']

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

def player_data_from_years(years, require_min_rounds=True, dict_by_id=True):
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
                    data[player_number] = {'name': player_name}

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

    drives = []
    putts_list = []

    # FOR EACH TOURNAMENT
    for tournament in tournament_file_dict:
        # Get the course data file
        print('=====================')
        print('Tournament %s' % (tournament))

        leaderboard_dict[tournament] = {}

        # FOR EACH YEAR
        for year in tournament_file_dict[tournament]:
            course_file_path = '/'.join(tournament_file_dict[tournament][year][0]['f'].split('/')[:-1]) + '/course.json'
            course_file = open(course_file_path, 'r')
            course_data = json.load(course_file)['courses'][0]
            course_file.close()
            course_name = '>> NO COURSE NAME <<'
            if course_data:
                course_name = course_data['name']
            print('Getting data from %s (%s)' % (course_name, year))

            # use this later to figure out rankings
            leaderboard_dict[tournament][year] = []

            # go through every player file this year
            for idx, player in enumerate(tournament_file_dict[tournament][year]):
                pid = player['id']
                print('\rGetting player %s' % (pid), end='')
                # if this player exists and has stats
                if pid in data and year in data[pid]:
                    file = open(tournament_file_dict[tournament][year][idx]['f'], 'r')

                    # scorecard data for this player for this tournament
                    scorecard_data = json.load(file)
                    file.close()

                    # get the player id
                    player = tournament_file_dict[tournament][year][idx]['id']

                    # the final dict we will populate
                    this_player_tournament_data = {}

                    # stick in the raw data to start - we'll also build a summary
                    this_player_tournament_data['scorecard'] = scorecard_data
                    summary = {}
                    rounds = scorecard_data['p']['rnds']

                    # the number of rounds this player played at this tournament
                    total_rounds = len(scorecard_data['p']['rnds'])

                    # scores on each hole (for the entire tournament, all N rounds)
                    hole_scores = []

                    # do we have valid data for this tournament
                    valid_data = True

                    for round in scorecard_data['p']['rnds']:
                        for hole in round['holes']:
                            if len(hole['sc']) == 0:
                                valid_data = False
                                break

                            hole_scores.append(int(hole['sc']))

                    # skip if data is incomplete
                    if valid_data == False:
                        continue

                    # skip if we have a weird number of rounds
                    if total_rounds not in [2,3,4]:
                        continue

                    # populate the easy stuff
                    summary['num_rounds'] =     total_rounds
                    summary['total_shots'] =    sum(hole_scores)
                    summary['course_name'] =    course_name
                    summary['course_yardage'] =  course_data['yards']
                    summary['course_par'] = course_data['parValue']
                    summary['hole_pars'] = []
                    summary['hole_yardages'] = []

                    round_stats = [{} for i in range(total_rounds)]

                    for idx, hole in enumerate(course_data['holes']):
                        try:
                            par = int(hole['parValue'].split(' / ')[0])
                        except:
                            par = 4
                        try:
                            yardage = int(hole['yards'].split(' / ')[0])
                        except:
                            yardage = 'nan'
                        summary['hole_pars'].append(str(par))
                        summary['hole_yardages'].append(str(yardage))

                        if 'holes' not in rounds[0] or len(rounds[0]['holes']) == 0:
                            continue

                        # print(total_rounds, rounds)

                        for round_num in range(total_rounds):
                            if 'score' not in round_stats[round_num]:
                                round_stats[round_num]['score'] = "0"

                            round_stats[round_num]['score'] = str(int(round_stats[round_num]['score']) + int(rounds[round_num]['holes'][idx]['sc']))

                            if 'drives' not in round_stats[round_num]:
                                round_stats[round_num]['drives'] = []
                            if 'putts_per_hole' not in round_stats[round_num]:
                                round_stats[round_num]['putts_per_hole'] = []

                            # SOMETIMES WE DON't HAVE INDIVIDUAL SHOT DATA
                            if par > 3:
                                if len(rounds[round_num]['holes'][idx]['shots']):
                                    drive = float(rounds[round_num]['holes'][idx]['shots'][0]['dist']) / 36.0
                                    if drive < 200:
                                        round_stats[round_num]['drives'].append('nan')
                                    else:
                                        round_stats[round_num]['drives'].append(str(drive))
                                else:
                                    round_stats[round_num]['drives'].append('nan')
                            else:
                                round_stats[round_num]['drives'].append('nan')

                            putts = 0
                            for shot in rounds[round_num]['holes'][idx]['shots']:
                                if len(shot['putt']):
                                    putts += 1
                            round_stats[round_num]['putts_per_hole'].append(str(putts))

                    for stats in round_stats:
                        if 'drives' in stats:
                            tot_drives = 0
                            num_drives = 0
                            for drive in stats['drives']:
                                if drive != 'nan':
                                    num_drives += 1
                                    tot_drives += float(drive)

                            if num_drives != 0:
                                stats['avg_drive'] = str(tot_drives / num_drives)
                                drives.append(tot_drives / num_drives)
                            else:
                                stats['avg_drive'] = 'nan'

                        if 'putts_per_hole' in stats:
                            tot_putts = 0
                            for putt in stats['putts_per_hole']:
                                tot_putts += int(putt)

                            if tot_putts > 0:
                                stats['total_putts'] = str(tot_putts)
                                putts_list.append(tot_putts)
                            else:
                                stats['total_putts'] = 'nan'
                                stats['putts_per_hole'] = ['nan' for p in stats['putts_per_hole']]

                    summary['round_stats'] = round_stats

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

    # plt.figure()
    # plt.hist(drives, bins=2000)
    # plt.figure()
    # plt.hist(putts_list, bins=20)
    # plt.show()

    return data

def stats_from_tournament(data, tournament_to_find, year, stat):
    stat_values = []
    for player in data:
        if year in data[player]:
            year_dict = data[player][year]
            if tournament_to_find in year_dict:
                if stat in year_dict['stats']:
                    stat_values.append(float(year_dict['stats'][stat]['value']))
    return stat_values

def summaries_from_tournament(data, tournament_to_find, year):
    scorecards = []
    for p in data:
        player = data[p]
        if year in player:
            year_data = player[year]

            for t in year_data:
                if t == 'stats': continue
                if t == tournament_to_find:
                    tournament = year_data[t]
                    scorecards.append(tournament['summary'])

    return scorecards

def select_from_summaries(summaries, vals):
    valid = np.array([int(s['num_rounds']) == 4 for s in summaries])
    np_scores = np.array(summaries)[valid]
    np_vals = np.array(vals)[valid]
    return (np_vals, np_scores)

def stats_and_scores(data, t, y, s):
    stats = stats_from_tournament(data, t, y, s)
    par = int(course_info_for_tournament(t, y)['course_par']) * 4
    summaries = [-(int(s['total_shots']) - par) for s in summaries_from_tournament(data, t, y)]
    return select_from_summaries(summaries, stats)

# extract info about the course for a particular tournament
def course_info_for_tournament(t, y):
    # build the cache if this is the first call (static function variable)
    if not hasattr(course_info_for_tournament, 'cache'):
        course_info_for_tournament.cache = {}
        years = ['2014', '2015', '2016']
        tournament_file_dict = tournament_files_for_years(years)

        # FOR EACH TOURNAMENT
        for tournament in tournament_file_dict:
            # FOR EACH YEAR
            for year in tournament_file_dict[tournament]:
                if year not in course_info_for_tournament.cache:
                    course_info_for_tournament.cache[year] = {}

                course_file_path = os.getcwd() + '/scorecards/' + tournament + '/' + year + '/course.json'

                try:
                    course_file = open(course_file_path, 'r')
                    course_data = json.load(course_file)['courses'][0]
                    course_file.close()
                except FileNotFoundError:
                    course_info_for_tournament.cache[year][tournament] = None
                    return None

                summary = {}

                pars = []
                yardages = []

                threes, fours, fives = [], [], []
                if 'holes' in course_data:
                    for hole in course_data['holes']:
                        pars.append(float(hole['parValue'].split(' / ')[0]))
                        yardages.append(hole['yards'].split(' / ')[0])

                        if int(pars[-1]) == 3:
                            threes.append(float(yardages[-1]))
                        elif int(pars[-1]) == 4:
                            fours.append(float(yardages[-1]))
                        else:
                            fives.append(float(yardages[-1]))


                # set the week in the schedule/name for this tournament
                summary['week'] =           course_data['week']
                summary['course_name'] = course_data['name']
                if 'yards' in course_data and len(course_data['yards']) > 1:
                    summary['course_yardage'] = float(re.sub('\D','',course_data['yards']))
                else:
                    summary['course_yardage'] = np.nan

                if 'parValue' in course_data and len(course_data['parValue']) > 1:
                    summary['course_par'] = float(course_data['parValue'])
                else:
                    summary['course_par'] = np.nan

                summary['hole_pars'] = pars
                summary['hole_yardages'] = yardages
                if len(threes) > 0:
                    summary['three_yardage'] = float(sum(threes) / len(threes))
                else:
                    summary['three_yardage'] = np.nan
                if len(fours) > 0:
                    summary['four_yardage'] = float(sum(fours) / len(fours))
                else:
                    summary['four_yardage'] = np.nan
                if len(fives) > 0:
                    summary['five_yardage'] = float(sum(fives) / len(fives))
                else:
                    summary['five_yardage'] = np.nan
                    
                # put our result in the cache
                course_info_for_tournament.cache[year][tournament] = summary

        for year in years:
            for tournament in course_info_for_tournament.cache[year]:
                # use this tournament's week as the default
                prev_week = -sys.maxsize

                this_week = int(course_info_for_tournament.cache[year][tournament]['week'])
                prev_ids = []

                for other_tournament in course_info_for_tournament.cache[year]:
                    other_week = int(course_info_for_tournament.cache[year][other_tournament]['week'])
                    # print('', other_week)
                    if other_week >= prev_week and other_week < this_week:
                        if other_week == prev_week:
                            prev_ids.append(other_tournament)
                        else:
                            prev_week = other_week
                            prev_ids = [other_tournament]
                if this_week - prev_week > 2: prev_ids = []
                course_info_for_tournament.cache[year][tournament]['prev_ids'] = prev_ids

    if t not in course_info_for_tournament.cache[y]: return None

    return course_info_for_tournament.cache[y][t]

# get the finish of a player the previous weekend
def rank_last_weekend(data, player_id, tournament, year):
    last_weekend_ids = course_info_for_tournament(tournament, year)['prev_ids']
    if len(last_weekend_ids) == 0: return np.nan
    
    for tournament_id in last_weekend_ids:
        if tournament_id in data[player_id][year]:
            t = data[player_id][year][tournament_id]['summary']
            return int(t['rank'])
            
    return np.nan

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
        start_year = 2014
        end_year = 2016
        data = player_data_from_years(
            [str(x) for x in range(start_year, end_year+1)], dict_by_id=False)

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

    min_val = sys.maxsize
    max_val = -sys.maxsize
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
    index_stats_in_data(reindex=True)

    # The years we want to look at
    #years = [str(y) for y in range(2013, 2017)]

    #data = player_data_from_years(years, dict_by_id=True)
    #scorecards_from_tournament(data, '010')
    # pprint(data)

    # show_stat_over_time(data, 'Putts Per Round', years)

    # stat_matrix, rank_matrix, index_as_stat, stat_names = gather(years)

    # print('Found %i records with %i unique stats' % stat_matrix.shape)
