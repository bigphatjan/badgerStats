import pandas as pd
import json
import urllib.request
from lifelines import NelsonAalenFitter
from numpy import inf,nan
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger()

#Check out https://lifelines.readthedocs.io/en/latest/Survival%20analysis%20with%20lifelines.html
#For some of the survival analysis plots

def load_data():
    """
    Loads Badger stat data from local data.json or batterseabadgers.co.uk

    Returns:
    dict: pandas dataframes for available categories

    As of Apr 2020 Badger stats are stored in the following tables:
    games(['slug', 'teamID', 'oppositionID', 'host', 'formatID', 'venueID','seasonID', 'date', 'timezone', 'result', 'margin', 'tosswonby','tossdecision', 'captainID', 'botmID', 'created_at', 'updated_at','deleted_at'])

    formats(['slug', 'name', 'innings', 'days', 'created_at', 'updated_at','deleted_at'])
    
    venues(['name', 'latitude', 'longitude', 'address', 'region', 'country','extra', 'slug', 'created_at', 'updated_at', 'deleted_at'])
    
    players(['slug', 'name', 'middlenames', 'surname', 'number', 'dob', 'batstyle','bowlstyle', 'height', 'image', 'created_at', 'updated_at','deleted_at'])
    
    innings(['gameID', 'teamID', 'innings_no', 'runs', 'wickets', 'overs','extras_noballs', 'extras_wides', 'extras_byes', 'extras_legbyes','extras_pens', 'declared', 'created_at', 'updated_at', 'deleted_at'])
    
    teams(['slug', 'name', 'shortname', 'website', 'created_at', 'updated_at','deleted_at'])
    
    battingPerformances(['gameID', 'inningsID', 'batnumber', 'playerID', 'runs', 'balls','fours', 'sixes', 'dismissal', 'bowlerID', 'fielderID','fielder_position', 'keeper', 'highlight', 'created_at', 'updated_at','deleted_at'])
    
    bowlingPerformances(['gameID', 'inningsID', 'playerID', 'balls', 'maidens', 'runs','wickets', 'wides', 'noballs', 'highlight', 'created_at', 'updated_at','deleted_at'])
    
    partnerships(['gameID', 'inningsID', 'bat1ID', 'bat2ID', 'wicketno', 'batoutID','runs', 'created_at', 'updated_at', 'deleted_at'])Index(['twitter_id', 'text', 'url', 'latitude', 'longitude','twitter_created_at', 'entity_urls', 'in_reply_to_status_id','created_at', 'updated_at', 'deleted_at'],type='object')
    
    awards(['player_id', 'category_id', 'season_id', 'game_id', 'description','created_at', 'updated_at', 'deleted_at'])
    
    awardCategories(['name', 'order', 'created_at', 'updated_at', 'deleted_at'], dtype='object')
    
    articles(['slug', 'title', 'content', 'banner', 'status', 'published','created_at', 'updated_at', 'deleted_at', 'game', 'banners','author.id', 'author.firstname', 'author.surname', 'author.email','author.status', 'author.updated_at', 'author.created_at','author.deleted_at'])

    For example:

    table=load_data()
    print(table['bowlingPerformances'].head()")
    """
    try:
        with open("./data.json") as response:
            data=json.load(response)
    except:
        logger.info("Couldn't find local Badger data - attempting to load from website")
        try:
            with urllib.request.urlopen("http://www.batterseabadgers.co.uk/data") as response:
                data=json.load(response)
                with open('data.json', 'w') as f:
                    json.dump(data, f)
        except Exception as e:
            raise e
    else:
        logger.info("Using local data.json - to use updated stats move this file and any pickled pandas dataframes")
    table={}
    for key in data['created']:
        table[key]=pd.io.json.json_normalize(data['created'][key])
        table[key]=table[key].set_index('id')
    return(table)

def process_data_for_quiz(table):
    quiz_players=[
        'andrew-thorpe',
        'jan-marchant',
        'martin-cloke',
        'peter-warman',
        'peter-jinks',
        'chris-shone',
        'joshua-lee',
        'nick-foord',
        'peter-cade',
        't-rex',
        'david-hirst',
        'james-hamblin'
    ]

    teams_abroad=[11,12,13,27,28,38,39,40,47,48,51,52,53,54,57,58,59,66,67,72,73,74,87,88,92]

    def getDataFromID(tableName,tableID,dataName):
        try:
            return table[tableName].loc[tableID][dataName]
        except:
            return None

    def getSlug(row):
        return getDataFromID('players',row['playerID'],'slug')

    def getFormatID(row):
        return getDataFromID('games',row['gameID'],'formatID')

    def getOppoID(row):
        return getDataFromID('games',row['gameID'],'oppositionID')

    def getTeamID(row):
        return getDataFromID('innings',row['inningsID'],'teamID')

    def isOut(row):
        return 0 if (row['dismissal']=='Not out' or row['dismissal'].startswith("Retired") or row['dismissal']=="Did not bat") else 1

    def isBadger(row):
        if (getTeamID(row)==1):
            return True
        else:
            return False

    def isOnTour(row,teams_abroad):
        if (getOppoID(row) in teams_abroad):
            return True
        else:
            return False

    def isQuizzer(row,quiz_players):
        if (getSlug(row) in quiz_players):
            return True
        else:
            return False

    bat="battingPerformances"
    bowl="bowlingPerformances"
    df1=(
        table[bat].assign(player=table[bat].apply(lambda x:getSlug(x),axis=1))
            .assign(out=table[bat].apply(lambda x:isOut(x),axis=1))
            .assign(formatID=table[bat].apply(lambda x:getFormatID(x),axis=1))
            .assign(isBadger=table[bat].apply(lambda x:isBadger(x),axis=1))
            .assign(onTour=table[bat].apply(lambda x:isOnTour(x,teams_abroad),axis=1))
            .assign(isQuizzer=table[bat].apply(lambda x:isQuizzer(x,quiz_players),axis=1))
    )

    df2=(
        table[bowl].assign(player=table[bowl].apply(lambda x:getSlug(x),axis=1))
            .assign(formatID=table[bowl].apply(lambda x:getFormatID(x),axis=1))
            .assign(isBadger=table[bowl].apply(lambda x:not isBadger(x),axis=1))
            .assign(onTour=table[bowl].apply(lambda x:isOnTour(x,teams_abroad),axis=1))
            .assign(isQuizzer=table[bowl].apply(lambda x:isQuizzer(x,quiz_players),axis=1))
    )

    df1.to_pickle("batting.pandas")
    df2.to_pickle("bowling.pandas")
    return(df1,df2)

def get_dataframes_for_quiz():
    try:
        df1=pd.read_pickle("batting.pandas")
        df2=pd.read_pickle("bowling.pandas")
    except:
        logger.info("Couldn't find pickled dataframes - loading might take a minute")
        try:
            table=load_data()
        except Exception as e:
            logger.info("Couldn't get data")
            logger.error(e)
            raise e
        (df1,df2)=process_data_for_quiz(table)
    return(df1,df2)

if __name__ == '__main__':
    #naughty Jan
    #Nelson-Aalen plot_hazard has some uncaught divide by zero warnings
    import warnings
    warnings.filterwarnings("ignore")

    naf = NelsonAalenFitter(0.05,False)

    try:
        (df1,df2)=get_dataframes_for_quiz()
    except Exception as e:
        logger.info("Couldn't load dataframes")
        logger.error(e)
        raise SystemExit

    def handle_question(question):
        if (question==1):
            #Not worth the hassle
            #print(table["formats"]["name"].sort_index())
            #format=int(input("Which format id? "))
            format=1
            print("The website is a bit cruel on some retirees so these numbers can be a little different:")
            print("===BATTING===")
            #N.B. Martin has a DNB from match 2 with 17 runs attached. Runs look spurious so not included.
            df=df1[(df1.isQuizzer) & (df1.isBadger) & (df1.formatID==format) & (df1.dismissal!="Did not bat")].groupby('player').sum()
            df['ave']=df['runs'].divide(df['out'])
            print(df[['runs','out','ave']].sort_values(by='ave',ascending=False))
            print("\n\n\n===BOWLING===")
            df=df2[(df2.isQuizzer) & (df2.isBadger) & (df2.formatID==format)].groupby('player').sum()
            df['ave']=df['runs'].divide(df['wickets'])
            print(df[['runs','wickets','ave']].sort_values(by='ave',ascending=True))
        if (question==2):
            innings=df1[(df1.dismissal!='Did not bat') & (df1.isBadger)]['player'].value_counts()
            print(innings[innings>99])
        if (question==3):
            n=0
            while n in df1[df1.isBadger].runs.values:
                n+=1
            print("Lowest batting score not recorded is {}.".format(n))
            n=0
            while n in df2[df2.isBadger].runs.values:
                n+=1
            print("Lowest bowling score not recorded is {}.".format(n))
        if (question==4):
            distance_per_run=17.68
            andy=df1[(df1.player=="andrew-thorpe")&(df1.isBadger)].groupby('player').sum()
            print("Andy has had to run {:.2f} km".format(((andy['runs']-andy['fours']*4-andy['sixes']*6)*distance_per_run/1000).iloc[0]))
        if (question==5):
            innings=df1[(df1.isBadger)]['player'].value_counts()
            not_outs=df1[(df1.dismissal=="Not out") & (df1.isBadger)]['player'].value_counts()
            df3=innings.to_frame().join(not_outs.to_frame(),lsuffix="_innings",rsuffix="_not_outs")[innings>29]
            df3['ratio']=df3["player_not_outs"].divide(df3["player_innings"])
            #df3.rename(columns={0: 'not out*'}, inplace=True)
            print(df3.sort_values(by='ratio',ascending=False))
        if (question==6):
            print("Josh Lee with 27")
        if (question==7):
            print("Chris Shone of course")
        if (question==8):
            innings=df1[(df1.isBadger)]['player'].value_counts()
            print('\n'.join(innings[innings==2].index.tolist()))
        if (question==9):
            print("====BATTING====")
            abroad=df1[df1.onTour].groupby('player').sum()
            home=df1[~df1.onTour].groupby('player').sum()

            abroad['ave']=abroad['runs'].divide(abroad['out'])
            home['ave']=home['runs'].divide(home['out'])

            dfbat=abroad.join(home,lsuffix='_abroad',rsuffix='_home')

            dfbat['ratio']=dfbat['ave_abroad'].divide(dfbat['ave_home'])

            print(dfbat[(dfbat.out_home>4) & (~dfbat.isin([nan, inf, -inf]).any(1))].sort_values(by='ratio',ascending=False)[['runs_abroad','out_abroad','ave_abroad','runs_home','out_home','ave_home','ratio']])
            print("\n\n====BOWLING====")
            abroad=df2[df2.onTour].groupby('player').sum()
            home=df2[~df2.onTour].groupby('player').sum()

            abroad['ave']=abroad['runs'].divide(abroad['wickets'])
            home['ave']=home['runs'].divide(home['wickets'])

            dfbowl=abroad.join(home,lsuffix='_abroad',rsuffix='_home')

            dfbowl['ratio']=dfbowl['ave_abroad'].divide(dfbowl['ave_home'])

            print(dfbowl[(dfbowl.wickets_home>4) & (~dfbowl.isin([nan, inf, -inf]).any(1))].sort_values(by='ratio')[['runs_abroad','wickets_abroad','ave_abroad','runs_home','wickets_home','ave_home','ratio']])
            print("\n\n====COMBINED====")
            dfboth=dfbat.join(dfbowl,lsuffix="_bat",rsuffix="_bowl")
            dfboth['overall']=dfboth['ratio_bat'].divide(dfboth['ratio_bowl'])
            print(dfboth[(dfboth.wickets_home>5) & (dfboth.out_home>5) & (~dfboth.isin([nan, inf, -inf]).any(1))].sort_values(by='overall',ascending=False)[['ratio_bat','ratio_bowl','overall']])
        if (question==10):
            T=df1[df1.isBadger & (df1.dismissal!='Did not bat')][['player','runs','out']]
            B=5
            naf.fit(T['runs'], T['out'],label="All Badgers")
            ax=naf.plot_hazard(ci_show=False,bandwidth=B)
            legend_list=[]
            legend_list.append("All Badgers")

            players=T['player'].unique()
            player=input("Which player? ")
            if (player not in players):
                print("Couldn't find {}. Choose one of the following:".format(player))
                print(' '.join(players))
            else:
                naf.fit(T[T.player==player]['runs'], T[T.player==player]['out'],label=player)
                naf.plot_hazard(ax=ax,ci_show=False,bandwidth=B)
                legend_list.append(player)
                plt.axvline(x=50,color="black")
                #plt.axvline(x=100,color="black")
                plt.xlim([0,75])
                plt.ylim([0,0.15])
                plt.xlabel("Runs")
                plt.ylabel("Hazard rate")
                ax.legend(legend_list,loc='upper center', bbox_to_anchor=(0.5, 1.1),fancybox=True, shadow=True, ncol=4)
                plt.show()
                #to save a pdf (can loop for as many players as you want)
                #plt.savefig('hazard.pdf',bbox_inches='tight')

    while True:

        try:
            question = int(input("Question (0 to quit): "))
        except ValueError:
            print("Error! This is not a number. Try again.")
        else:
            if (question==0):
                break
            else:
                handle_question(question)

