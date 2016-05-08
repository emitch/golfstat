from bs4 import BeautifulSoup as Soup
from scipy.stats import spearmanr
import requests, json, os, parsedata
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, LinearRegression

def scrape_scorecards():
    """ mine the pgatour data endpoint for individual tournament scorecards
    and copy the .json files and folder structure to the disk """

    # format for scorecard url
    source_url = 'http://www.pgatour.com/data/r/'

    # parse base page to get each tournament
    source_soup = Soup(requests.get(source_url).text, 'html.parser')

    # open each linked tournament and continue searching
    for tourn in source_soup.find_all('a'):
        # skip non-tournament links
        tourn_path = tourn.get('href')
        if not tourn_path.strip('/').isdigit(): continue
        # parse tournament page and follow all 'year' links
        tourn_url = source_url + tourn_path
        tourn_soup = Soup(requests.get(tourn_url).text, 'html.parser')

        # follow each link, hoping for a scorecards folder
        for year in tourn_soup.find_all('a'):
            # skip non year links
            year_path = year.get('href')
            if not year_path.strip('/').isdigit(): continue
            if len(year_path) != 5: continue
            # get page response
            score_url = tourn_url + year_path + 'scorecards/'
            response = requests.get(score_url)
            # skip nonexistant scorecards
            if response.status_code == 404: continue
            # parse the page
            score_soup = Soup(response.text, 'html.parser')
            # locate each json page
            for link in score_soup.find_all('a'):
                # skip .xml versions
                linked_file = link.get('href')
                if not linked_file.endswith('.json'): continue
                # open json page
                data = requests.get(score_url + linked_file).text

                # create folder structure
                tourn_folder = 'scorecards/' + tourn_path
                if not os.path.exists(tourn_folder):
                    os.mkdir(tourn_folder)
                destination = tourn_folder + year_path
                if not os.path.exists(destination):
                    os.mkdir(destination)

                # save
                with open(destination + linked_file, 'w') as file:
                    file.write(data)
                # output data
                print('Saved %s\r' % (destination + linked_file), end='')

def scrape_courses():
    """ get the par and distance information for each course used in the
    tournaments with scorecard data.  scrape scorecards must be run first
    as we use the existing local folder structure to build the urls """
    # pga tour data endpoint
    baseurl = 'http://www.pgatour.com/data/r/'
    # iterate through tournament folders
    for tourn in os.listdir('scorecards/'):
        # skip hidden folders
        if tourn.startswith('.'): continue
        for year in os.listdir('scorecards/{0}/'.format(tourn)):
            # skip hidden folders
            if year.startswith('.'): continue
            # current location in folder structure
            path = tourn + '/' + year + '/course.json'
            print('Scraping %s\r' % path, end='')
            # concatenate with baseurl to get endpoint
            data = requests.get(baseurl + path).text
            with open('scorecards/' + path, 'w') as file: file.write(data)

def players_in_tourn(tourn_id, year):
    """ return an iterable of player ids corresponding to tournament
    participation """
    # navigate to tournament folder
    path = 'scorecards/{0}/'.format(tourn_id)
    # initialize list of players
    players = []
    for file in os.listdir(path + year):
        # add each player id to the list
        player_id = file.strip('.json')
        if player_id.isdigit(): players.append(player_id)

    return players


def course_stat_bias(data, tourn_id, stat_as_index, years=None):
    """ Calculate the spearman correlation of each stat with tournament
    performance on a given course, indicating how important each stat
    is for success at a tournament """
    # get years for tournament
    if years is None:
        years = [x for x in os.listdir('scorecards/' + tourn_id) if x.isdigit()]
    # get scores for each player
    scores = []
    for year in years:
        # CHECK IF IT EXISTS THO
        if parsedata.course_info_for_tournament(tourn_id, year) is None: continue
        players = players_in_tourn(tourn_id, year)
        for player in players:
            # skip missing players
            if player not in data: continue
            if year not in data[player]: continue
            if tourn_id not in data[player][year]: continue

            # compile performance and stuff
            shots = data[player][year][tourn_id]['summary']['total_shots']
            rnds = data[player][year][tourn_id]['summary']['num_rounds']

            scores.append(float(shots)/float(rnds))

    # initialize vector of spearman correlations
    corrs = np.empty(len(stat_as_index))
    for stat in stat_as_index:
        # vector of stat for each player
        stats = []
        for year in years:
            if parsedata.course_info_for_tournament(tourn_id, year) is None: continue
            players = players_in_tourn(tourn_id, year)
            for player in players:
                # skip missing players
                if player not in data: continue
                if year not in data[player]: continue
                if tourn_id not in data[player][year]: continue
                # compile performance and stuff
                try: val = parse_stat(data[player][year]['stats'][stat]['value'])
                except KeyError: val = np.nan
                stats.append(val)

        # calculate correlation
        corrs[stat_as_index[stat]], _ = spearmanr(scores, stats)

    return corrs

def parse_stat(raw_val):
    # parse stat to float, taking account of wonky formatting
    if '%' in raw_val: val = float(raw_val[:-1])/100
    elif '$' in raw_val: val = float(raw_val[1:])
    elif "'" in raw_val:
        # convert distance to inches
        split = raw_val.split("' ")
        val = int(split[0]) * 12 + int(split[1][:-1])
    else:
        try: val = float(raw_val)
        except: val = np.nan
    return val

###############################################################################
# METHODS FOR BUILDING A FEATURE VECTOR
# ANY OF THESE CAN BE PASSED TO build_fvs()
###############################################################################

def basic_fv(player, course, biases, stat_as_index):
    # multiply stats by their bias
    n_stats = len(stat_as_index)
    fv = np.empty(n_stats)
    for stat in stat_as_index:
        try: val = parse_stat(player['stats'][stat]['value'])
        except KeyError: val = np.nan
        fv[stat_as_index[stat]] = biases[stat_as_index[stat]] * val
        #print('Bias: % 5f \t Value: % .3f \t(%s)' % (biases[stat_as_index[stat]], val, stat))

    return fv

def include_distance(player, course, biases, stat_as_index):
    # basic_fv but also append driving ratio and stuff
    basic = basic_fv(player, course, biases, stat_as_index)
    n_stats = len(basic)
    fv = np.empty(n_stats + 3)
    # get each other stat
    try: dist = float(player['stats']['Driving Distance']['value'])
    except KeyError: dist = np.nan

    fv[:-3] = basic
    fv[-3] = float(course['three_yardage']) / dist
    fv[-2] = float(course['four_yardage']) / dist
    fv[-1] = float(course['five_yardage']) / dist

    return fv

def include_last_tourn(player, course, biases, stat_as_index):
     # basic_fv but also append driving ratio and stuff
    basic = basic_fv(player, course, biases, stat_as_index)
    n_stats = len(basic)
    fv = np.empty(n_stats + 1)
    fv[:-1] = basic
    fv[-1] =

###############################################################################
# END OF FEATURE VECTOR BUILDING
###############################################################################

def build_fvs(data, year, stat_as_index, make_vector=basic_fv):
    """ train a model to predict performance on the level of tournaments
    using individual player stats and their importance for each course.
    The function make_vector is the one that builds a feature vector from
    the input data: data, biases, and stat_as_index """
    # First gather all the course biases, final feature vectors are player stats
    # modulated by the degree to which a given stat predicts success on a course
    biases = {}
    for tourn_id in os.listdir('scorecards/'):
        if not tourn_id.isdigit(): continue
        biases[tourn_id] = course_stat_bias(data, tourn_id, stat_as_index, [year])

    # now compile feature vectors, iterating through players and looking up
    # course biases for each tournament, also get tournament data
    feature_vector_list = []
    scores = []
    for player in data:
        if year not in data[player]: continue
        for tourn in data[player][year]:
            if not tourn.isdigit(): continue
            # get any additional course data
            course_info = parsedata.course_info_for_tournament(tourn, year)
            # make feature vectors
            fv = make_vector(
                data[player][year], course_info, biases[tourn], stat_as_index)
            # add to running list
            feature_vector_list.append(fv)
            # get results
            shots = data[player][year][tourn]['summary']['total_shots']
            rnds = data[player][year][tourn]['summary']['num_rounds']

            par = float(course_info['course_par'])
            scores.append((float(shots)/float(rnds))-par)

    # concatenate list into single np matrix
    unbiased_stats = np.vstack(feature_vector_list)
    scores = np.array(scores)

    return unbiased_stats, scores

# def hole_stat_bias(data, tourn_id, years, h):
#     raise
#     """ Calculate the spearman correlation of each stat with tournament
#     performance on a given hole, indicating how important each stat
#     is for success on a hole """
#     # get years for tournament
#     if years is None:
#         years = [x for x in os.listdir('scorecards/' + tourn_id) if x.isdigit()]
#     # get scores for each player
#     scores = []
#     for year in years:
#         players = players_in_tourn(tourn_id, year)
#         for player in players:
#             # skip missing players
#             if player not in data: continue
#             if year not in data[player]: continue
#             if tourn_id not in data[player][year]: continue

#             # compile performance and stuff
#             score = data[player][year][tourn_id]['scorecard']['h'][h-1]['sc']

#             scores.append()

#     # initialize vector of spearman correlations
#     corrs = np.empty(len(stat_as_index))
#     for stat in stat_as_index:
#         # vector of stat for each player
#         stats = []
#         for year in years:
#             players = players_in_tourn(tourn_id, year)
#             for player in players:
#                 # skip missing players
#                 if player not in data: continue
#                 if year not in data[player]: continue
#                 if tourn_id not in data[player][year]: continue
#                 # compile performance and stuff
#                 try: val = parse_stat(data[player][year]['stats'][stat]['value'])
#                 except KeyError: val = np.nan
#                 stats.append(val)

#         # calculate correlation
#         corrs[stat_as_index[stat]], _ = spearmanr(scores, stats)

#     return corrs, info

def test_model(data, stat_as_index, make_vector, regressor):
    # compile and shit
    print('Compiling stats...')
    fv, sc = [], []
    for year in ['2014', '2015', '2016']:
        f,s = build_fvs(
            data, year, stat_as_index, make_vector=make_vector)
        fv.append(f)
        sc.append(s)

    # # train model and evaluate using kfold cross validation
    # print('Training model...')
    # n_obs = len(sc)
    # pred = np.empty(n_obs)
    # for train, test in KFold(n_obs, 100):
    #     # fit
    #     mod = regressor
    #     mod.fit(bs[train,:], sc[train])
    #     # predict
    #     pred[test] = mod.predict(bs[test,:])

    # Compile into single vectors: Predict 2016 from 2014 and 2015
    fv_train, fv_test = np.vstack(fv[0:2]), fv[2]
    sc_train, sc_test = np.concatenate(sc[0:2]), sc[2]

    train_nan = np.isnan(fv_train)
    test_nan = np.isnan(fv_test)

    # Impute NaNs
    print('Imputing')
    if train_nan.any():
        i1 = Imputer()
        fv_train = i1.fit_transform(fv_train)
        print(i1.statistics_)
    if test_nan.any():
        i2 = Imputer()
        fv_test = i2.fit_transform(fv_test)
        print(i2.statistics_)

    # Exclude players with missing scores
    train_nan, test_nan = np.isnan(sc_train), np.isnan(sc_test)
    fv_train, sc_train = fv_train[~train_nan], sc_train[~train_nan]
    fv_test, sc_test = fv_test[~test_nan], sc_test[~test_nan]

    # Build model
    mod = regressor
    mod.fit(fv_train, sc_train)
    pred = mod.predict(fv_test)

    # average errors
    print('Calculating errors...')
    err = pred - sc_test
    rmse = np.nanmean(err ** 2) ** .5
    print('Root-mean-square-error: %.4f' % rmse)

    return rmse, mod

if __name__ == '__main__':
    # load data
    data = parsedata.player_data_from_years(
        ['2014', '2015', '2016'], dict_by_id=True)
    with open('stat_as_index.json', 'r') as f:
        stat_as_index = json.load(f)

    # compile and shit
    print('Basic FV, Lasso')
    test_model(data, stat_as_index, basic_fv, LassoCV())

    # try again with whitelist
    whitelist = ['Birdie Average', 'Scrambling', 'Scrambling from the Rough',
        'Scoring Average', 'Sand Save Percentage', 'Driving Distance']

    wl = {}
    for i, stat in enumerate(whitelist): wl[stat] = i
    print('Whitelisted stats')
    test_model(data, wl, basic_fv, LinearRegression())

    print('POOP')
    test_model(data, wl, include_distance, LinearRegression())

