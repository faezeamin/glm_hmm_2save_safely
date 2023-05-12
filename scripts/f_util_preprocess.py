import numpy as np
import pandas as pd
import os


def get_path_to_data_dir():
    path_to_file = '/Users/faezeamin/Documents/Allen/ssm_forkedFromLinderman_Cloned/ForkedFromSL/data/' #+ mouse_name +'.npy'
    return(path_to_file)

def get_path_to_save_dir():
    path_to_save = '/Users/faezeamin/Documents/Allen/ssm_forkedFromLinderman_Cloned/ForkedFromSL/results/'
    return(path_to_save)

def import_data(path_to_file):
    data_array = np.load(path_to_file, allow_pickle = True)
    data = pd.DataFrame(data_array.tolist())
    return data

def n_input_dim(hparams):
    if hparams['analysis_experiment_name'] == 'just_bias':
        input_dim = 1

    if hparams['analysis_experiment_name'] == 'bias_RewardOneHot':
        input_dim = 1 + 2 * hparams['num_history_step']

    if hparams['analysis_experiment_name'] == 'bias_RewardOneHot_ChoiceOneHot':
        input_dim = 1 + 4 * hparams['num_history_step']

    if hparams['analysis_experiment_name'] == 'modified_Bari':
        input_dim = 1 + 3 * hparams['num_history_step']

    return input_dim


def generate_hyperparameter_sets_for_mle_map(alphas,sigmas):
    hyperparameter_sets =[]

    for alpha in alphas:
        for sigma in sigmas:
            hyperparameter_sets.append([alpha,sigma])

    return(hyperparameter_sets)


def get_list_of_trial_count_in_chosen_sessens(hparams, data):
    n_trials_in_sessions = []
    for sess in hparams['sessions']:
        n_trials_in_sessions.append(np.shape(data[sess]['choice_history'])[1])
    return n_trials_in_sessions

def get_num_of_trial_count_in_chosen_sessen(data, sess):
    n_trials_in_the_session = np.shape(data[sess]['choice_history'])[1]
    return n_trials_in_the_session 


def get_input_choice_array_for_selected_sessions_in_serries(hparams, data):
    if hparams['GLM'] == 'multinomial':
        """
        if hparams['GLM'] == 'multinomial'
            output: numpyarray size:[total number of trials across all the selected sessionns , 2]

            choice_array will be a 2-dim array where
            first column:    right : choicen [1] / not-chosen [0], 
            second column:   left  : choicen [1] / not-chosen [0]
        """

        inpt_choice_array = np.zeros((sum(hparams['list_of_trial_count_in_chosen_sessens']) , 2))
        trial_count = 0
        for i,sess in enumerate(hparams['sessions']):
            trial_n_sess = hparams['list_of_trial_count_in_chosen_sessens'][i]
            for trial in range(trial_n_sess):
                if data[sess]['choice_history'][0,trial] == 1: inpt_choice_array[trial + trial_count,0] = 1 #right choice
                if data[sess]['choice_history'][0,trial] == 0: inpt_choice_array[trial + trial_count,1] = 1 #left choice
            trial_count += trial_n_sess

    if hparams['GLM'] == 'binomial':
        pass    

    return inpt_choice_array




def get_n_trials_for_input_array(hparams):
    """
    This function calculates total number of trials to be used in the input array (design matrix parts)
    n_trials = n_tot_trials + (-n_hist_step +1) * n_sess
    """
    n_sess = len(hparams['list_of_trial_count_in_chosen_sessens'])
    n_trials_for_input_array = sum(hparams['list_of_trial_count_in_chosen_sessens']) + (-hparams['num_history_step']) * n_sess

    return n_trials_for_input_array


def get_block_divider_in_selected_sess_rowwise(hparams,data):
    """This function outputs a list of size (n_sess). for each sess, we have an array of block_dividers_in_the_sess. 
    output: list_blocks - list :    size (n_sess)
                                    each element of list: size (1, n_blocks_in_corresponding_sess)
    """
    list_blocks = []
    for sess_id,sess in enumerate(hparams['sessions']):
        cumulative_trila_num = np.array(np.where(np.diff(data[sess]['p_reward'][0]) != 0)[0])
        cumulative_trila_num = np.append(cumulative_trila_num, hparams['num_input_trials_segr_by_sessions'][sess_id])
        cumulative_trila_num[0] = cumulative_trila_num[0]-hparams['num_history_step']
        hparams['num_input_trials_segr_by_sessions'][sess_id]
        list_blocks.append(cumulative_trila_num)

    return list_blocks


def get_block_divider_in_selected_sess(hparams,data):
    """This function outputs a list of size (1). sess are arranged in one row. trials that marks block_dividers_across_all_sess). 
    output: list - size (1, n_blocks_in_all_sess)"""

    list_blocks = get_block_divider_in_selected_sess_rowwise(hparams,data)
    list_blocks_concat = []
    for i in range(len(list_blocks)):
        list_blocks_concat.extend(list_blocks[i])
        if i != len(list_blocks)-1: list_blocks[i+1] = list_blocks[i+1] + list_blocks_concat[-1]
    return list_blocks_concat


def get_sess_divider_in_selected_sess(hparams):
    """This function outputs a list of size (1). sess are arranged in one row. trials that marks sess_dividers_across_all_sess). 
    output: list - size (1, n_sess_in_all_sess)"""
    sess_devider_trials = np.cumsum(np.array(hparams['num_input_trials_segr_by_sessions'])).tolist()
    return sess_devider_trials 


def get_sess_of_trial(hparams, trial_num):
    """This function takes the trial number in the selected sessions, which are placed in one row, ad returnns its corresponding session"""
    import bisect
    sess_divider = get_sess_divider_in_selected_sess(hparams)
    index = bisect.bisect_right(sess_divider, trial_num)
    corresp_sess = hparams['sessions'][index]
    return corresp_sess


def get_list_of_num_trials_for_chosen_sess_in_input_array(hparams):
    """
    This function calculates the number of trials to be used in the input array segregated by the sesseions (design matrix parts)
    
    output :    list size : (1,num_sess) ; 
                n_trials_coress_to each_sess = n_tot_trials - n_hist_step
    """
    num_input_trials_in_sessions = (np.array(hparams['list_of_trial_count_in_chosen_sessens']) -hparams['num_history_step']).tolist()  
    return num_input_trials_in_sessions


def build_input_array_for_n_history_steps(hparams,data, kw ):
    """
    This fuction builds choice_array of n_step_history for model input from concatenated session data
    kw: choose from ['choice_input', 'reward_input']

    output: numpyarray size:[total number of trials across all the selected sessions + (-num_history_step + 1) * n_sess , 2*num_history_step]
            input_array_for_n_hist_step can be input_choice_array or input_reward_array. look at kw description.
            The built array has considered the begining of each sessions to contribuite independently than the previous sess.

    input_choice_array_for_n_hist_step is a 2-dim array where:
            first column:        (n)_step before -      right : choicen [1] / not-chosen [0], 
            second column:       (n)_step before -      left  : choicen [1] / not-chosen [0],

            third column:        (n-1)_step before -    right : choicen [1] / not-chosen [0], 
            forth column:        (n-1)_step before -    left  : choicen [1] / not-chosen [0],

            ...

            one_to_last column:  (1)_step before -      right : choicen [1] / not-chosen [0], 
            last column:         (1)_step before -      left  : choicen [1] / not-chosen [0],

    """
    if kw == 'choice_input': base_input_array = get_input_choice_array_for_selected_sessions_in_serries(hparams, data)
    if kw == 'reward_input': base_input_array = get_input_reward_array_for_selected_sessions_in_serries(hparams, data)
    num_history_step = hparams['num_history_step']
    list_of_trial_count_in_chosen_sessens = hparams['list_of_trial_count_in_chosen_sessens']

    input_array_for_n_hist_step = np.zeros(( hparams['num_trials_in_input_array'] , 2*num_history_step ))
    tot_num_trials_countered_for_input_array = 0
    tot_num_experiment_trials_considered = 0

    for n_trial_in_the_sess in list_of_trial_count_in_chosen_sessens:
        n_trial_to_consider_for_the_sess = n_trial_in_the_sess - num_history_step
        
        for step in range(num_history_step):
            input_array_for_n_hist_step[tot_num_trials_countered_for_input_array : n_trial_to_consider_for_the_sess + tot_num_trials_countered_for_input_array, 2*step : 2*(step+1)] = \
                base_input_array[tot_num_experiment_trials_considered + num_history_step-1-step : tot_num_experiment_trials_considered + n_trial_in_the_sess -step-1 , :]

        tot_num_trials_countered_for_input_array += n_trial_to_consider_for_the_sess
        tot_num_experiment_trials_considered += n_trial_in_the_sess

    return input_array_for_n_hist_step


def reward_array_in_shape_trial_by_LR(arr):
    """ 
    This function is meant to use to change the form of reward to (trial , 2). It swaps rows 0 and 1, then transposes it.
    input: arr size(tial , N)
    output: np.array - size(N, tial) 
        first dim:          trials
        second dim:         side(right/left)    first column:    right : rewarded [1] / not-rewarded [0], 
                                                second column:   left  : rewarded [1] / not-rewarded [0]
     """
    arr [[0,1]] = arr[[1,0]]
    arr = arr.T
    return arr


def get_input_reward_array_for_selected_sessions_in_serries(hparams, data):
    """
    This function biuld a reward matrix for all the selected sessions in serires in the shape of (trial , 2)
    output: numpyarray size:[total number of trials across all the selected sessionns , 2]
        first dim:          trials
        second dim:         side(right/left)    first column:    right : rewarded [1] / not-rewarded [0], 
                                                second column:   left  : rewarded [1] / not-rewarded [0]
    """
    inpt_reward_array = np.zeros((sum(hparams['list_of_trial_count_in_chosen_sessens']) , 2))
    trial_count = 0
    for i,sess in enumerate(hparams['sessions']):
        rew_arr = np.copy(data[sess]['reward_history'])
        rew_arr = reward_array_in_shape_trial_by_LR(rew_arr)
        trial_num_in_sess = hparams['list_of_trial_count_in_chosen_sessens'][i]
        inpt_reward_array[trial_count: trial_num_in_sess + trial_count, :] = rew_arr
        trial_count += trial_num_in_sess

    return inpt_reward_array


def build_input_bias_array_for_selected_sessions(hparams):
    """
    This function biuld the bias array considering all the trials in the selected sessions 
    output: 1-D numpyarray - 
            size:[total number of trials across all the selected sessions + (- num_history_step +1 )* n_sess , 1]
    """
    inpt_bias_array = np.ones((hparams['num_trials_in_input_array'],1))
    return inpt_bias_array


def get_input_choice_modifiedBari_encoding_for_one_session(hparams, data, sess):
    if hparams['GLM'] == 'multinomial':
        """
        if hparams['GLM'] == 'multinomial'
            output: numpyarray size:[total number of trials for the selected sessionns , 2]
                    first colum encodes "right and left" choice, second column encodes "Ignored" choice

            choice_array will be a 1-dim array where
            first column:   right choicen           [1]  
                            left choicen            [-1]
                            otherwise(ignored)      [0],
            second column:  ignored                 [1]
                            otherwise               [0]
        """
        n_trials_in_the_session = get_num_of_trial_count_in_chosen_sessen(data, sess)
        inpt_choice_array_sess = np.zeros((n_trials_in_the_session , 2))
        choice_arr_sess = np.copy(data[sess]['choice_history'][0])

        choice_arr_sess[choice_arr_sess == 0] = -1
        choice_arr_sess[np.isnan(choice_arr_sess)] = 0
        inpt_choice_array_sess[:,0] = choice_arr_sess
        inpt_choice_array_sess[np.where(choice_arr_sess == 0) ,1] = 1 


    if hparams['GLM'] == 'binomial':
        pass    

    return inpt_choice_array_sess


def expand_input_feature_arr_to_n_history_steps_feature_arr_for_one_session(hparams,feature_arr):  
    """
    This fuction takes the feature_arr of size [num_trial_in-the-sess , 1] and builds feature_array of n_step_history for model input for a single session

    output:  numpyarray size:[total number of trials of the selected session + -num_history_step + 1   , 2*num_history_step]
    Input:   feature_arr - size [num_trial_in-the-sess , 1]
             can be any feature array, input_choiceRightLeft, input_choiceNull, input_reward_array. 
    """
    num_history_step = hparams['num_history_step']
    
    arr_history = np.zeros((feature_arr.shape[0]-num_history_step, num_history_step))

    for i in range(arr_history.shape[0]):
        arr_history[i] = feature_arr[i:i+num_history_step].T

    return arr_history


def build_input_choice_array_for_n_history_steps_Right1LeftMinus1Otherwise0_encoding_across_all_sessions(hparams,data): 
    """Right1LeftMinus1Otherwise0_encoding: (can be used in Bari encoding, ....):
                    choice:                 right : [1]
                                            left : [-1]
                                            otherwise : [0]
        output: input array of choice - size (num_trials_in_input_array'] ,  num_hist_step):
            """
    num_hist_step = hparams['num_history_step']
    inpt_choice_RL_all_sess = np.empty( (hparams['num_trials_in_input_array'] ,  num_hist_step) )
    i = 0
    for sess_i , sess in enumerate(hparams['sessions']):
        inpt_choice_array_one_sess = get_input_choice_modifiedBari_encoding_for_one_session(hparams, data, sess)
        
        inp_choice_RightLeft_one_sess_n_hist = expand_input_feature_arr_to_n_history_steps_feature_arr_for_one_session(hparams,inpt_choice_array_one_sess[:,0])

        input_len_for_sess = hparams['list_of_trial_count_in_chosen_sessens'][sess_i] - num_hist_step

        inpt_choice_RL_all_sess[i : i + input_len_for_sess ,:] = inp_choice_RightLeft_one_sess_n_hist

        i += input_len_for_sess

    return inpt_choice_RL_all_sess
    

def build_input_choice_array_for_n_history_steps_Ignored1Otherwise0_encoding_across_all_sessions(hparams,data): 
    """Right1LeftMinus1Otherwise0_encoding: (can be used in Bari encoding, ....):
                    choice:                  Ignored : [1]
                                             otherwise : [0]
        output: input array of choice - size (num_trials_in_input_array'] ,  num_hist_step):
            """
    num_hist_step = hparams['num_history_step']
    inpt_choice_ignored_all_sess = np.empty( (hparams['num_trials_in_input_array'] ,  num_hist_step) )
    i = 0
    for sess_i , sess in enumerate(hparams['sessions']):
        inpt_choice_array_one_sess = get_input_choice_modifiedBari_encoding_for_one_session(hparams, data, sess)
        
        inp_choice_ignored_one_sess_n_hist = expand_input_feature_arr_to_n_history_steps_feature_arr_for_one_session(hparams,inpt_choice_array_one_sess[:,1])

        input_len_for_sess = hparams['list_of_trial_count_in_chosen_sessens'][sess_i] - num_hist_step

        inpt_choice_ignored_all_sess[i : i + input_len_for_sess ,:] = inp_choice_ignored_one_sess_n_hist

        i += input_len_for_sess

    return inpt_choice_ignored_all_sess
    

def build_input_choice_array_for_n_history_steps_all_sess_modifiedBari_encoding(hparams,data): 
    """Bari encoding:
                    choice: first colums:  right : [1]
                                            left : [-1]
                                            otherwise : [0]
                            second column:  Ignored : [1]
                                            otherwise : [0]
        output: size (num_trials_in_input_array, 2*num_hist_steps)
            """
    
    inpt_choice_RL_all_sess = build_input_choice_array_for_n_history_steps_Right1LeftMinus1Otherwise0_encoding_across_all_sessions(hparams,data)
    inpt_choice_ignored_all_sess = build_input_choice_array_for_n_history_steps_Ignored1Otherwise0_encoding_across_all_sessions(hparams,data)

    inpt_choice_part = np.hstack((inpt_choice_RL_all_sess, inpt_choice_ignored_all_sess))

    return inpt_choice_part

 
def get_input_reward_array_one_session_RightRew1LeftRewMinus1Otherwise0_encoding(data, sess):
    """
    data[]['reward_history']: 2-d array, first dimension: left [0] / right [1], second dimension: trial number. 0 = no reward, 1 = rewarded

    output: numpyarray size:[total number of trials for the selected sessionns , 1]
            reward_array will be a 1-dim array where
            column:     right rewarded           [1]  
                        left rewarded            [-1]
                        otherwise (no reward)    [0]
    """
    rew_arr = np.copy(data[sess]['reward_history'])

    inpt_reward_array_sess = rew_arr[1,:]
    inpt_reward_array_sess[rew_arr[0,:] == 1] = -1

    return inpt_reward_array_sess.reshape(-1,1)


def build_input_reward_array_for_n_history_steps_all_sess_Right1LeftMinus1Otherwise0_encoding(hparams,data): 
    """Right1LeftMinus1Otherwise0_encoding: (can be used in Bari encoding, ....):
                    reward:                 right : [1]
                                            left : [-1]
                                            otherwise : [0]
        output: input array of choice - size (num_trials_in_input_array'] ,  num_hist_step):
            """
    num_hist_step = hparams['num_history_step']
    inpt_reward_RL_all_sess = np.zeros( (hparams['num_trials_in_input_array'] ,  num_hist_step) )
    i = 0
    for sess_i , sess in enumerate(hparams['sessions']):
        inpt_reward_array_one_sess = get_input_reward_array_one_session_RightRew1LeftRewMinus1Otherwise0_encoding(data, sess)
        
        inp_reward_Right1LeftMinus1Otherwise0_one_sess_n_hist = expand_input_feature_arr_to_n_history_steps_feature_arr_for_one_session(hparams,inpt_reward_array_one_sess)

        input_len_for_sess = hparams['list_of_trial_count_in_chosen_sessens'][sess_i] - num_hist_step

        inpt_reward_RL_all_sess[i : i + input_len_for_sess ,:] = inp_reward_Right1LeftMinus1Otherwise0_one_sess_n_hist

        i += input_len_for_sess

    return inpt_reward_RL_all_sess



def build_input_design_matrix(hparams, data):
    """
    This fucntion builds the input design matrix
    output: list - size: (1, n_input_trials, n_iput_dim) 
    """
    
    input_design_matrix_list = []

    if hparams['analysis_experiment_name'] == 'just_bias':
        input_design_matrix = build_input_bias_array_for_selected_sessions(hparams)

    if hparams['analysis_experiment_name'] == 'bias_RewardOneHot':
        inpt_bias_part = build_input_bias_array_for_selected_sessions(hparams)
        inpt_reward_part = build_input_array_for_n_history_steps(hparams,data, kw = 'reward_input' )
        input_design_matrix = np.hstack((inpt_bias_part, inpt_reward_part))

    if hparams['analysis_experiment_name'] == 'bias_RewardOneHot_ChoiceOneHot':
        inpt_bias_part = build_input_bias_array_for_selected_sessions(hparams)
        inpt_reward_part = build_input_array_for_n_history_steps(hparams,data, kw = 'reward_input' )
        inpt_choice_part = build_input_array_for_n_history_steps(hparams,data, kw = 'choice_input' )
        input_design_matrix = np.hstack((inpt_bias_part, inpt_reward_part, inpt_choice_part))

    if hparams['analysis_experiment_name'] == 'modified_Bari':
        inpt_bias_part = build_input_bias_array_for_selected_sessions(hparams)
        inpt_reward_part =  build_input_reward_array_for_n_history_steps_all_sess_Right1LeftMinus1Otherwise0_encoding(hparams,data)
        inpt_choice_part = build_input_choice_array_for_n_history_steps_all_sess_modifiedBari_encoding(hparams,data)
        input_design_matrix = np.hstack((inpt_bias_part, inpt_reward_part, inpt_choice_part))

    input_design_matrix_list.append(input_design_matrix)
    return input_design_matrix_list


def get_input_labels(hparams):
    reward_labels, choice_labels = [], []
    choice_RL_labels, choice_ignrd_labels = [], []

    if hparams['GLM'] == 'multinomial':

        if hparams['analysis_experiment_name'] == 'just_bias':
            input_labels = ['bias']


        if hparams['analysis_experiment_name'] == 'bias_RewardOneHot':
            for k in range(hparams['num_history_step'], 0, -1):
                reward_labels.append('Rew_{j}_Right'.format(j=k))
                reward_labels.append('Rew_{j}_Left'.format(j=k))

            list_of_input_labels = [['bias'], reward_labels]
            input_labels = [elem for sublist in list_of_input_labels for elem in sublist]


        if hparams['analysis_experiment_name'] == 'bias_RewardOneHot_ChoiceOneHot':
            for k in range(hparams['num_history_step'], 0, -1):
                reward_labels.append('Rew_{j}_Right'.format(j=k))
                reward_labels.append('Rew_{j}_Left'.format(j=k))
                choice_labels.append('Ch_{j}_Right'.format(j=k))
                choice_labels.append('Ch_{j}_Left'.format(j=k))

            list_of_input_labels = [['bias'], reward_labels, choice_labels]
            input_labels = [elem for sublist in list_of_input_labels for elem in sublist]


        if hparams['analysis_experiment_name'] == 'modified_Bari': 
            for k in range(hparams['num_history_step'], 0, -1):
                reward_labels.append('Rew_{j}'.format(j=k))
                choice_RL_labels.append('Ch_RL_{j}'.format(j=k))
                choice_ignrd_labels.append('Ch_Ignord_{j}'.format(j=k))

            list_of_input_labels = [['bias'], reward_labels, choice_RL_labels, choice_ignrd_labels]
            input_labels = [elem for sublist in list_of_input_labels for elem in sublist]



        # to be completed

    return input_labels


def get_encoded_output_choice(data_choice, hparams):
    """
    This function set the output choice array in the form of : 0 = right, 1 = left, 2 = nan (ignored)

    input : 'choice_history' :  1-d numpy_array - size(n_trials_in_the_sess, 1) - raw data
                                0 = left, 1 = right, nan = ignored
             kw can be chosen from ['R0_L1_nan2' , 'R0_nan1_L2']

    output: 'choice_history' :  
                                1-d numpy_array - size(n_trials_in_the_sess, 1)
                                if kw = 'R0_L1_nan2'
                                    0 = right, 1 = left, 2 = nan (ignored)
                                if kw = 'R0_nan1_L2'
                                    0 = right, 1 = nan (ignored), 2 = left

    """
    arr = data_choice.copy()

    if hparams['output_form'] == 'R0_L1_nan2':
        arr[arr == 0] = 3
        arr[arr == 1] = 0
        arr[arr == 3] = 1
        arr[np.isnan(arr)] = 2

    if hparams['output_form'] == 'R0_nan1_L2':
        arr[arr == 0] = 3
        arr[arr == 1] = 0
        arr[arr == 3] = 2 
        arr[np.isnan(arr)] = 1  

    if hparams['output_form'] =='otherwise':
        pass
        # maybe to be completed
        
    arr = arr.astype(np.int64)
    return arr


def build_output_choice_list(hparams, data):
    """
    This fucntion builds the output choice list for the GLM-HMM model
    kw : refer to function "get_encoded_output_choice(data_choice, kw)"

    output: list - size: (1, total number of trials across all the selected sessions - num_history_step * n_sess, 1) 
    """
    output_choice_list = []

    n_trials = hparams['num_trials_in_input_array']
    output_array = np.zeros((n_trials , 1),dtype=np.int64)
    trial_count = 0

    for i,sess in enumerate(hparams['sessions']):
        data_choice = get_encoded_output_choice(data[sess]['choice_history'], hparams).T

        n_trials_in_the_sess_to_consider =  hparams['list_of_trial_count_in_chosen_sessens'][i] - hparams['num_history_step']
        output_array[trial_count:trial_count + n_trials_in_the_sess_to_consider ,0]  = data_choice[hparams['num_history_step']:, 0]

        trial_count += n_trials_in_the_sess_to_consider

    output_choice_list.append(output_array.copy())

    return output_choice_list



def get_output_class_list(hparams):
    import re
    str = hparams['output_form']
    elements = re.findall('[RL]|nan', str)
    replace_dict = {
    'R': '"Right"',
    'L': '"Left"',
    'nan': '"Ignored"'
    }
    output_list = [replace_dict.get(x, x) for x in elements]
    return output_list


def permute(arr, i,j):
    """define a function to replace the specified elements in an array with new values"""
    new_arr = arr.copy()
    new_arr[arr == i] = j
    new_arr[arr == j] = i
    return new_arr


def count_matches(arr1, arr2):
    """define a function to compute the number of matching elements between two arrays"""
    return np.sum(arr1 == arr2)


def permute_array(arr1, arr2):
    """try all possible replacements of 0s with 1s, or 0s with 2s, or 1s with 2s"""
    best_count = count_matches(arr1, arr2)

    n_unique_elem = len(np.unique(arr2))
    for i in range(n_unique_elem):
        for j in range(n_unique_elem):
            if i != j:
                new_arr = permute(arr2, i,j)
                count = count_matches(arr1, new_arr)
                if count >= best_count:
                    best_count = count
                    best_i = i
                    best_j = j
                    print(f"Best replacement: {i}{j}, Count: {best_count}")

    if best_i: 
        new_arr = permute(arr2, best_i,best_j)
        return new_arr
    else:
        return arr2
    


def get_frac_occ_across_sess(hparams, infered_states):
    """
    input: infered_states - size (1,num_inputs)
    output: np.array - size (num_states, num_sess) - each colums is frac_occ of the corresponding session
    """
    past_divider_tri = 0
    num_states = len(np.unique(infered_states))
    frac_occupancies_per_sess = np.zeros((num_states, len(hparams['sess_divider_trials'])))

    for i, divider_tri in enumerate(hparams['sess_divider_trials']):
        states_in_sess = infered_states[past_divider_tri:divider_tri]

        # obtain state fractional occupancies:
        frac_occ = np.zeros((num_states,))
        for k in range(num_states):
            frac_occ[k] = np.count_nonzero(states_in_sess == k)
        frac_occ = frac_occ/ np.sum(frac_occ)
        frac_occupancies_per_sess[:,i] = frac_occ.copy()

        past_divider_tri = divider_tri

    return frac_occupancies_per_sess



def generate_synthetic_choice(hparams, true_choices):
    # choice encoding : 0 = right, 1 = left, 2 = nan (ignored)
    p_base = np.zeros((3,))
    true_choice_base =[]
    for i in range(hparams['num_categories']):
        p_base[i] = np.sum(np.array(true_choices).squeeze()==i) / hparams['num_trials_in_input_array']

    categories = [0, 1, 2]

    # Sample from the categorical distribution
    choice_base = np.random.choice(categories, size = hparams['num_trials_in_input_array'], p=p_base)
    true_choice_base.append(choice_base.reshape(-1,1))

    return true_choice_base