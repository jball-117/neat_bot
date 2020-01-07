## TAKES IN REPLAY FILES FROM AND GIVES DF ##
import carball
import pandas as pd
from numpy import isnan
from google.protobuf.json_format import MessageToDict
from tqdm import tqdm
import os

DELETE_WHEN_LEFT = 1

pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

## converting replays ##
#rootdir = '/home/zach/Files/Nas/Replays'
rootdir = '/home/zach/Files/Nas/ReplayModels/ReplayDataProcessing/RANKED_STANDARD/Replays/1400-1600'
for root, dirs, files in os.walk(rootdir):
    for filename in tqdm(files):
        if not filename.endswith('.replay'):
            print("\n", filename, "not a replay\n")
            continue
        csv_name = rootdir+"/CSVs/"+filename
        csv_name = csv_name.replace('.replay', '.csv')
        if os.path.exists(csv_name):
            print("\n", csv_name, "exists\n")
            continue
        print("ANALYZING...")
        try:
            analysis_manager = carball.analyze_replay_file(os.path.abspath(os.path.join(root, filename)))
        except Exception as e:
            print("ERROR WITH REPLAY ANALYSIS\n", e)
            os.remove(os.path.abspath(os.path.join(root, filename)))
            continue
        proto_game = analysis_manager.get_protobuf_data()
        df = analysis_manager.get_data_frame()
        df = df.astype('float64')
        df.reset_index(drop=True, inplace=True)
        dict_game = MessageToDict(proto_game)

        player_team = {}
        best_score = 0
        know_score = True
        know_team = True
        for i in dict_game['players']:
            # indentifies MVP
            try:
                if i['score'] > best_score:
                    best_score = i['score']
            except KeyError:
                know_score = False
                print("NO SCORE")
                break
            try:
                if i['isOrange']:
                    player_team.update({i['name']: tuple([i['score'], 'orange'])})
                else:
                    player_team.update({i['name']: tuple([i['score'], 'blue'])})
            except KeyError:
                know_team = False
                print("NO TEAM")
                break
        if not know_team or not know_score:
            continue

        # identifying best player(s)
        ordered_playas = []
        for name, score in player_team.items():
            if score[0] == best_score:
                ordered_playas.append(name)

        playas = list(list(df.columns.levels)[0])
        playas.remove('ball')
        playas.remove('game')

        for playa in playas:
            if playa not in ordered_playas and player_team[playa][1] == player_team[ordered_playas[0]][1]:
                ordered_playas.append(playa)

        for playa in playas:
            if playa not in ordered_playas and player_team[playa][1] != player_team[ordered_playas[0]][1]:
                ordered_playas.append(playa)


        ########################### MAKING SURE EACH LEVEL IS SAME LENGTH
        length = len(df['ball'])
        if length != len(df['game']):
            print("BAD")
        for playa in ordered_playas:
            if len(df[playa]) != length:
                print("BAD")
        ###########################

        ## MAKING NEW SINGLE LEVEL DF FOR TRAINING ##
        ## PLAYER 0, 1, AND 1 ARE HIGHEST SCORING PLAYER'S TEAM WHILE 3, 4, AND 5 ARE ON OPPOSITE TEAM ##
        player_desired = 0
        single_level_df = df[ordered_playas[player_desired]]
        #################################################################################################################
        '''
        rem_cols = ['rot_x','rot_y','rot_z','vel_x','vel_y','vel_z','ang_vel_x','ang_vel_y',\
                'ang_vel_z','ping','throttle','steer','handbrake','ball_cam','boost','boost_active','jump_active',\
                'double_jump_active','dodge_active','boost_collect']  
        for col in rem_cols:
            if col in single_level_df.columns:
                single_level_df.drop(columns=[col], inplace=True)
        single_level_df.rename(columns={'pos_x': str(player_desired)+'_pos_x', 'pos_y': str(player_desired)+'_pos_y', 'pos_z': str(player_desired)+'_pos_z'}, inplace=True)
        '''
        single_level_df.drop(columns=['ping'], inplace=True)
        single_level_df.rename(columns={'pos_x': str(player_desired)+'_pos_x', 'pos_y': str(player_desired)+'_pos_y', 'pos_z': str(player_desired)+'_pos_z', 
                'rot_x': str(player_desired)+'_rot_x', 'rot_y': str(player_desired)+'_rot_y', 'rot_z': str(player_desired)+'_rot_z', 
                'vel_x': str(player_desired)+'_vel_x', 'vel_y': str(player_desired)+'_vel_y', 'vel_z': str(player_desired)+'_vel_z', 
                'ang_vel_x': str(player_desired)+'_ang_vel_x', 'ang_vel_y': str(player_desired)+'_ang_vel_y', 'ang_vel_z': str(player_desired)+'_ang_vel_z',
                'throttle': str(player_desired)+'_throttle', 'steer': str(player_desired)+'_steer', 'handbrake': str(player_desired)+'_handbrake',
                'ball_cam': str(player_desired)+'_ball_cam', 'boost': str(player_desired)+'_boost', 'boost_active': str(player_desired)+'_boost_active', 
                'jump_active': str(player_desired)+'_jump_active', 'double_jump_active': str(player_desired)+'_double_jump_active', 'dodge_active': str(player_desired)+'_dodge_active', 
                'boost_collect': str(player_desired)+'_boost_collect'}, inplace=True)
        #################################################################################################################
        for i, playa in enumerate(ordered_playas):
            if player_desired == i:
                continue
            piece = df[playa]
            #################################################################################################################
            '''
            rem_cols = ['ping','throttle','steer','handbrake','ball_cam','boost','boost_active','jump_active',\
                    'double_jump_active','dodge_active','boost_collect']
            for col in rem_cols:
                if col in piece.columns:
                    piece.drop(columns=[col], inplace=True)
            piece.rename(columns={'pos_x': str(i)+'_pos_x', 'pos_y': str(i)+'_pos_y', 'pos_z': str(i)+'_pos_z', \
                    'rot_x': str(i)+'_rot_x', 'rot_y': str(i)+'_rot_y', 'rot_z': str(i)+'_rot_z', 'vel_x': str(i)+'_vel_x', \
                    'vel_y': str(i)+'_vel_y', 'vel_z': str(i)+'_vel_z', 'ang_vel_x': str(i)+'_ang_vel_x', 'ang_vel_y': str(i)+'_ang_vel_y', \
                    'ang_vel_z': str(i)+'_ang_vel_z'}, inplace=True)
            '''
            piece.drop(columns=['ping'], inplace=True)
            piece.rename(columns={'pos_x': str(i)+'_pos_x', 'pos_y': str(i)+'_pos_y', 'pos_z': str(i)+'_pos_z', 
                    'rot_x': str(i)+'_rot_x', 'rot_y': str(i)+'_rot_y', 'rot_z': str(i)+'_rot_z', 
                    'vel_x': str(i)+'_vel_x', 'vel_y': str(i)+'_vel_y', 'vel_z': str(i)+'_vel_z', 
                    'ang_vel_x': str(i)+'_ang_vel_x', 'ang_vel_y': str(i)+'_ang_vel_y', 'ang_vel_z': str(i)+'_ang_vel_z',
                    'throttle': str(i)+'_throttle', 'steer': str(i)+'_steer', 'handbrake': str(i)+'_handbrake',
                    'ball_cam': str(i)+'_ball_cam', 'boost': str(i)+'_boost', 'boost_active': str(i)+'_boost_active', 
                    'jump_active': str(i)+'_jump_active', 'double_jump_active': str(i)+'_double_jump_active', 'dodge_active': str(i)+'_dodge_active', 
                    'boost_collect': str(i)+'_boost_collect'}, inplace=True)
            #################################################################################################################
            single_level_df = single_level_df.join(piece)
            
        ball_data = df['ball']
        ball_data.drop(columns=['hit_team_no'], inplace=True)
        ball_data.rename(columns={'pos_x': 'ball_pos_x', 'pos_y': 'ball_pos_y', 'pos_z': 'ball_pos_z', 'rot_x': 'ball_rot_x', \
                'rot_y': 'ball_rot_y', 'rot_z': 'ball_rot_z', 'vel_x': 'ball_vel_x', 'vel_y': 'ball_vel_y', 'vel_z': 'ball_vel_z', \
                'ang_vel_x': 'ball_ang_vel_x', 'ang_vel_y': 'ball_ang_vel_y', 'ang_vel_z': 'ball_ang_vel_z'}, inplace=True)
        single_level_df = single_level_df.join(ball_data)
        single_level_df['seconds_remaining'] = df['game']['seconds_remaining']
        
        # need to do this if someone leaves game
        # THIS IS STILL BROKEN. THIS ONLY CHECKS ONE ROW FOR ALL NAN
        # MAKE SURE EVERYTHING ELSE FOR THAT PLAYER AFTER THAT ROW IS NAN
        if not all(k in single_level_df.columns for k in ['0_pos_x', '0_pos_y', '0_pos_z', '1_pos_x', '1_pos_y', '1_pos_z',
            '2_pos_x', '2_pos_y', '2_pos_z', '3_pos_x', '3_pos_y', '3_pos_z', '4_pos_x', '4_pos_y', '4_pos_z',
            '5_pos_x', '5_pos_y', '5_pos_z']):
            print("not all players accounted for")
            continue
        single_level_df.reset_index(drop=True, inplace=True)
        if DELETE_WHEN_LEFT:
            print("CHECKING IF PLAYERS LEFT...")
            rem_rows = set()
            for i in single_level_df.index:
                if (isnan(single_level_df.at[i, '0_pos_x']) and isnan(single_level_df.at[i, '0_pos_y']) and isnan(single_level_df.at[i, '0_pos_z'])):
                    #and isnan(single_level_df.at[i, '0_rot_x']) and isnan(single_level_df.at[i, '0_rot_y']) and isnan(single_level_df.at[i, '0_rot_z']) \
                    #and isnan(single_level_df.at[i, '0_vel_x']) and isnan(single_level_df.at[i, '0_vel_y']) and isnan(single_level_df.at[i, '0_vel_z']) \
                    #and isnan(single_level_df.at[i, '0_ang_vel_x']) and isnan(single_level_df.at[i, '0_ang_vel_y']) and isnan(single_level_df.at[i, '0_ang_vel_z']) \
                    #and isnan(single_level_df.at[i, '0_throttle']) and isnan(single_level_df.at[i, '0_steer']) and isnan(single_level_df.at[i, '0_handbrake']) \
                    #and isnan(single_level_df.at[i, '0_ball_cam']) and isnan(single_level_df.at[i, '0_boost']) and isnan(single_level_df.at[i, '0_boost_active']) \
                    #and isnan(single_level_df.at[i, '0_jump_active']) and isnan(single_level_df.at[i, '0_double_jump_active']) and isnan(single_level_df.at[i, '0_dodge_active']) \
                    #and isnan(single_level_df.at[i, '0_boost_collect'])):
                    '''
                    player_columns = []
                    for x in single_level_df.columns:
                        if x.startswith('0'):
                            player_columns.append(x)
                    '''
                    tmp = single_level_df[i:]
                    #tmp = tmp.filter(items=player_columns)
                    tmp = tmp.filter(items=['0_pos_x', '0_pos_y', '0_pos_z'])
                    if tmp.dropna(how='all').empty:
                        print('player 0 left the game')
                        single_level_df = single_level_df[:i]
                        break
                    else:
                        rem_rows.add(i)
                if (isnan(single_level_df.at[i, '1_pos_x']) and isnan(single_level_df.at[i, '1_pos_y']) and isnan(single_level_df.at[i, '1_pos_z'])):
                    #and isnan(single_level_df.at[i, '1_rot_x']) and isnan(single_level_df.at[i, '1_rot_y']) and isnan(single_level_df.at[i, '1_rot_z']) \
                    #and isnan(single_level_df.at[i, '1_vel_x']) and isnan(single_level_df.at[i, '1_vel_y']) and isnan(single_level_df.at[i, '1_vel_z']) \
                    #and isnan(single_level_df.at[i, '1_ang_vel_x']) and isnan(single_level_df.at[i, '1_ang_vel_y']) and isnan(single_level_df.at[i, '1_ang_vel_z']) \
                    #and isnan(single_level_df.at[i, '1_throttle']) and isnan(single_level_df.at[i, '1_steer']) and isnan(single_level_df.at[i, '1_handbrake']) \
                    #and isnan(single_level_df.at[i, '1_ball_cam']) and isnan(single_level_df.at[i, '1_boost']) and isnan(single_level_df.at[i, '1_boost_active']) \
                    #and isnan(single_level_df.at[i, '1_jump_active']) and isnan(single_level_df.at[i, '1_double_jump_active']) and isnan(single_level_df.at[i, '1_dodge_active']) \
                    #and isnan(single_level_df.at[i, '1_boost_collect'])):
                    '''
                    player_columns = []
                    for x in single_level_df.columns:
                        if x.startswith('1'):
                            player_columns.append(x)
                    '''
                    tmp = single_level_df[i:]
                    #tmp = tmp.filter(items=player_columns)
                    tmp = tmp.filter(items=['1_pos_x', '1_pos_y', '1_pos_z'])
                    if tmp.dropna(how='all').empty:
                        print('player 1 left the game')
                        single_level_df = single_level_df[:i]
                        break
                    else:
                        rem_rows.add(i)
                if (isnan(single_level_df.at[i, '2_pos_x']) and isnan(single_level_df.at[i, '2_pos_y']) and isnan(single_level_df.at[i, '2_pos_z'])):
                    #and isnan(single_level_df.at[i, '2_rot_x']) and isnan(single_level_df.at[i, '2_rot_y']) and isnan(single_level_df.at[i, '2_rot_z']) \
                    #and isnan(single_level_df.at[i, '2_vel_x']) and isnan(single_level_df.at[i, '2_vel_y']) and isnan(single_level_df.at[i, '2_vel_z']) \
                    #and isnan(single_level_df.at[i, '2_ang_vel_x']) and isnan(single_level_df.at[i, '2_ang_vel_y']) and isnan(single_level_df.at[i, '2_ang_vel_z']) \
                    #and isnan(single_level_df.at[i, '2_throttle']) and isnan(single_level_df.at[i, '2_steer']) and isnan(single_level_df.at[i, '2_handbrake']) \
                    #and isnan(single_level_df.at[i, '2_ball_cam']) and isnan(single_level_df.at[i, '2_boost']) and isnan(single_level_df.at[i, '2_boost_active']) \
                    #and isnan(single_level_df.at[i, '2_jump_active']) and isnan(single_level_df.at[i, '2_double_jump_active']) and isnan(single_level_df.at[i, '2_dodge_active']) \
                    #and isnan(single_level_df.at[i, '2_boost_collect'])):
                    '''
                    player_columns = []
                    for x in single_level_df.columns:
                        if x.startswith('2'):
                            player_columns.append(x)
                    '''
                    tmp = single_level_df[i:]
                    #tmp = tmp.filter(items=player_columns)
                    tmp = tmp.filter(items=['2_pos_x', '2_pos_y', '2_pos_z'])
                    if tmp.dropna(how='all').empty:
                        print('player 2 left the game')
                        single_level_df = single_level_df[:i]
                        break
                    else:
                        rem_rows.add(i)
                if (isnan(single_level_df.at[i, '3_pos_x']) and isnan(single_level_df.at[i, '3_pos_y']) and isnan(single_level_df.at[i, '3_pos_z'])):
                    #and isnan(single_level_df.at[i, '3_rot_x']) and isnan(single_level_df.at[i, '3_rot_y']) and isnan(single_level_df.at[i, '3_rot_z']) \
                    #and isnan(single_level_df.at[i, '3_vel_x']) and isnan(single_level_df.at[i, '3_vel_y']) and isnan(single_level_df.at[i, '3_vel_z']) \
                    #and isnan(single_level_df.at[i, '3_ang_vel_x']) and isnan(single_level_df.at[i, '3_ang_vel_y']) and isnan(single_level_df.at[i, '3_ang_vel_z']) \
                    #and isnan(single_level_df.at[i, '3_throttle']) and isnan(single_level_df.at[i, '3_steer']) and isnan(single_level_df.at[i, '3_handbrake']) \
                    #and isnan(single_level_df.at[i, '3_ball_cam']) and isnan(single_level_df.at[i, '3_boost']) and isnan(single_level_df.at[i, '3_boost_active']) \
                    #and isnan(single_level_df.at[i, '3_jump_active']) and isnan(single_level_df.at[i, '3_double_jump_active']) and isnan(single_level_df.at[i, '3_dodge_active']) \
                    #and isnan(single_level_df.at[i, '3_boost_collect'])):
                    '''
                    player_columns = []
                    for x in single_level_df.columns:
                        if x.startswith('3'):
                            player_columns.append(x)
                    '''
                    tmp = single_level_df[i:]
                    #tmp = tmp.filter(items=player_columns)
                    tmp = tmp.filter(items=['3_pos_x', '3_pos_y', '3_pos_z'])
                    if tmp.dropna(how='all').empty:
                        print('player 3 left the game')
                        single_level_df = single_level_df[:i]
                        break
                    else:
                        rem_rows.add(i)
                if (isnan(single_level_df.at[i, '4_pos_x']) and isnan(single_level_df.at[i, '4_pos_y']) and isnan(single_level_df.at[i, '4_pos_z'])):
                    #and isnan(single_level_df.at[i, '4_rot_x']) and isnan(single_level_df.at[i, '4_rot_y']) and isnan(single_level_df.at[i, '4_rot_z']) \
                    #and isnan(single_level_df.at[i, '4_vel_x']) and isnan(single_level_df.at[i, '4_vel_y']) and isnan(single_level_df.at[i, '4_vel_z']) \
                    #and isnan(single_level_df.at[i, '4_ang_vel_x']) and isnan(single_level_df.at[i, '4_ang_vel_y']) and isnan(single_level_df.at[i, '4_ang_vel_z']) \
                    #and isnan(single_level_df.at[i, '4_throttle']) and isnan(single_level_df.at[i, '4_steer']) and isnan(single_level_df.at[i, '4_handbrake']) \
                    #and isnan(single_level_df.at[i, '4_ball_cam']) and isnan(single_level_df.at[i, '4_boost']) and isnan(single_level_df.at[i, '4_boost_active']) \
                    #and isnan(single_level_df.at[i, '4_jump_active']) and isnan(single_level_df.at[i, '4_double_jump_active']) and isnan(single_level_df.at[i, '4_dodge_active']) \
                    #and isnan(single_level_df.at[i, '4_boost_collect'])):
                    '''
                    player_columns = []
                    for x in single_level_df.columns:
                        if x.startswith('4'):
                            player_columns.append(x)
                    '''
                    tmp = single_level_df[i:]
                    #tmp = tmp.filter(items=player_columns)
                    tmp = tmp.filter(items=['4_pos_x', '4_pos_y', '4_pos_z'])
                    if tmp.dropna(how='all').empty:
                        print('player 4 left the game')
                        single_level_df = single_level_df[:i]
                        break
                    else:
                        rem_rows.add(i)
                if (isnan(single_level_df.at[i, '5_pos_x']) and isnan(single_level_df.at[i, '5_pos_y']) and isnan(single_level_df.at[i, '5_pos_z'])):
                    #and isnan(single_level_df.at[i, '5_rot_x']) and isnan(single_level_df.at[i, '5_rot_y']) and isnan(single_level_df.at[i, '5_rot_z']) \
                    #and isnan(single_level_df.at[i, '5_vel_x']) and isnan(single_level_df.at[i, '5_vel_y']) and isnan(single_level_df.at[i, '5_vel_z']) \
                    #and isnan(single_level_df.at[i, '5_ang_vel_x']) and isnan(single_level_df.at[i, '5_ang_vel_y']) and isnan(single_level_df.at[i, '5_ang_vel_z']) \
                    #and isnan(single_level_df.at[i, '5_throttle']) and isnan(single_level_df.at[i, '5_steer']) and isnan(single_level_df.at[i, '5_handbrake']) \
                    #and isnan(single_level_df.at[i, '5_ball_cam']) and isnan(single_level_df.at[i, '5_boost']) and isnan(single_level_df.at[i, '5_boost_active']) \
                    #and isnan(single_level_df.at[i, '5_jump_active']) and isnan(single_level_df.at[i, '5_double_jump_active']) and isnan(single_level_df.at[i, '5_dodge_active']) \
                    #and isnan(single_level_df.at[i, '5_boost_collect'])):
                    '''
                    player_columns = []
                    for x in single_level_df.columns:
                        if x.startswith('5'):
                            player_columns.append(x)
                    '''
                    tmp = single_level_df[i:]
                    #tmp = tmp.filter(items=player_columns)
                    tmp = tmp.filter(items=['5_pos_x', '5_pos_y', '5_pos_z'])
                    if tmp.dropna(how='all').empty:
                        print('player 5 left the game')
                        single_level_df = single_level_df[:i]
                        break
                    else:
                        rem_rows.add(i)

        single_level_df.drop(rem_rows, errors='ignore', inplace=True)
        if not single_level_df.empty:
            print("WRITING", csv_name)
            single_level_df.to_csv(csv_name)
    break
