from bs4 import BeautifulSoup as Soup
from scipy.stats import spearmanr
import requests, json, os, parsedata
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, f1_score

###############################################################################
# SCRAPING METHODS: RUN ONCE AND NEVER AGAIN
###############################################################################

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

###############################################################################
# END OF SCRAPING METHODS
###############################################################################

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
    fv[-1] = parsedata.rank_last_weekend(player, course)

    return fv

###############################################################################
# END OF FEATURE VECTOR BUILDING
###############################################################################

def compute_biases(data, stat_as_index, year):
    biases = {}
    for tourn_id in os.listdir('scorecards/'):
        if not tourn_id.isdigit(): continue
        biases[tourn_id] = course_stat_bias(data, tourn_id, stat_as_index, [year])

    return biases

def build_fvs(data, year, stat_as_index, make_vector=basic_fv, target='score'):
    """ make features to predict performance on the level of tournaments
    using individual player stats and their importance for each course.
    The function make_vector is the one that builds a feature vector from
    the input data: data, biases, and stat_as_index """
    # First gather all the course biases, final feature vectors are player stats
    # modulated by the degree to which a given stat predicts success on a course
    biases = compute_biases(data, stat_as_index, year)

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
            if target == 'score':
                # get results
                shots = data[player][year][tourn]['summary']['total_shots']
                rnds = data[player][year][tourn]['summary']['num_rounds']

                par = float(course_info['course_par'])
                scores.append((float(shots)/float(rnds))-par)
            elif target == 'cut':
                if data[player][year][tourn]['summary']['rank'] == 'mc':
                    scores.append(False)
                else: scores.append(True)
            else: raise ValueError('Bad target')


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

def test_model(data, stat_as_index, make_vector, model, do_pca=False, target='score'):
    # compile and shit
    print('Compiling stats...')
    fv, sc = [], []
    for year in ['2014', '2015', '2016']:
        f,s = build_fvs(
            data, year, stat_as_index, make_vector, target)
        fv.append(f)
        sc.append(s)

    # Compile into single vectors: Predict 2016 from 2014 and 2015
    fv_train, fv_test = np.vstack(fv[0:2]), fv[2]
    sc_train, sc_test = np.concatenate(sc[0:2]), sc[2]

    # Impute NaNs
    train_nan = np.isnan(fv_train)
    test_nan = np.isnan(fv_test)
    
    for i in range(fv_train.shape[1]):
        if np.isnan(fv_train[0,i]):
            fv_train[0,i] = 0
    for i in range(fv_test.shape[1]):
        if np.isnan(fv_test[0,i]):
            fv_test[0,i] = 0
            
    print('Imputing...')
    if train_nan.any():
        i1 = Imputer()
        fv_train = i1.fit_transform(fv_train)
        #print(i1.statistics_)
    if test_nan.any():
        i2 = Imputer()
        fv_test = i2.fit_transform(fv_test)
        #print(i2.statistics_)

    if do_pca:
        pca = PCA(whiten=True)
        fv_train = pca.fit_transform(fv_train)
        fv_test = pca.transform(fv_test)

    # Exclude players with missing scores
    train_nan, test_nan = np.isnan(sc_train), np.isnan(sc_test)
    fv_train, sc_train = fv_train[~train_nan], sc_train[~train_nan]
    fv_test, sc_test = fv_test[~test_nan], sc_test[~test_nan]

    # Build model
    mod = model
    mod.fit(fv_train, sc_train)
    # kluge to allow for classifier and regressor evaluation
    try: pred = mod.predict_proba(fv_test)
    except: pred = mod.predict(fv_test)

    return pred, sc_test, mod

def rmse(pred, real):
    return np.nanmean((pred - real) ** 2) ** .5

def f1(real, pred):
    pred = pred > 0.5
    return f1_score(real, pred)

def show_bias(data, year, stat_as_index, stats_to_show, tourns_to_show, tourn_names=None):
    # get biases
    biases = compute_biases(data, stat_as_index, year)

    n_groups = len(tourns_to_show)
    n_stats = len(stats_to_show)
    bar_width = 0.9 / n_stats
    colors = ['c','m','y','k','g']

    # plot those motherfuckers
    fig = plt.figure()
    for i, tourn in enumerate(tourns_to_show):
        for j, stat in enumerate(stats_to_show):
            offset = j*bar_width
            plt.bar(i + 0.5 + offset, abs(biases[tourn][stat_as_index[stat]]),
                bar_width, color=colors[j%(n_stats)])

    if tourn_names is None: tourn_names = tourns_to_show

    plt.xticks(range(1,len(tourns_to_show)+1), tourn_names)
    plt.ylabel('Spearman correlation of stat with score')
    plt.legend(stats_to_show, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3)
    beautify(fig)

    plt.show()

def beautify(fig):
    # Set background color to white
    fig.patch.set_facecolor((1,1,1))

    # Remove frame and unnecessary tick-marks
    ax = plt.subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tick_params(axis='y', which='both', left='off', right='off')
    plt.tick_params(axis='x', which='both', top='off', bottom='off')
    return ax

if __name__ == '__main__':
    # load data
    data = parsedata.player_data_from_years(
        ['2014', '2015', '2016'], dict_by_id=True)
    with open('stat_as_index.json', 'r') as f:
        stat_as_index = json.load(f)

    # stats = ['Driving Distance', 'Scrambling', 'Sand Save Percentage',
    #     'Overall Putting Average', 'Proximity to Hole']
    # tourns = ['010', '012', '013']
    # names = ['Honda Classic', 'RBC Heritage', 'Wyndham Championship']
    # show_bias(data, '2015', stat_as_index, stats, tourns, names)

    # compile and shit
    # pred, real, _ = test_model(data, stat_as_index, include_last_tourn, GaussianNB(), do_pca=False, target='cut')
    # print(f1(real, pred[:,1]), roc_auc_score(real, pred[:,1]), 'GNB, all features')
    
    pred, real, _ = test_model(data, stat_as_index, include_last_tourn, LassoCV(cv=100), do_pca=False)
    err = pred - real
    rmse = np.nanmean(err ** 2) ** .5
    print(rmse)

    # try again with whitelist
    results = {}
    for whitelist in os.listdir('whitelists'):
        if whitelist.startswith('.'): continue
        wl = {}
        with open('whitelists/' + whitelist) as file:
            for i, line in enumerate(file): wl[line.strip('\n')] = i

        pred, real, _ = test_model(
            data, wl, include_last_tourn, GaussianNB(), do_pca=False, target='cut')
        results[whitelist.strip('.csv')] = (f1(real, pred[:,1]), roc_auc_score(real, pred[:,1]))

    for name in results:
        print(results[name], name)
