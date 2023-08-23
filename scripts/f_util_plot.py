import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from f_util_preprocess import *
from f_util_analyses import *


def set_color_pallete():
    cols =[]
    for i in sns.color_palette("tab10"):
        cols.append(mpl.colors.rgb2hex(i))
    return cols


def plot_heatmap_of_pearson_corr_coeff_of_input_data(hparams, input_design_matrix_list, threshold = 0.5):
    corr_ = get_pearson_corr_coeff_of_input_data(input_design_matrix_list)
    np.fill_diagonal(corr_, 0)
    data = corr_


    # create a copy of the data array and set values below the threshold to NaN
    data_threshold = data.copy()
    data_threshold[np.abs(data) < threshold] = np.nan

    # create the heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(data_threshold, cmap="seismic", vmin=-1, vmax=1)

    # loop over the data and add text annotations
    for i in range(len(data)):
        for j in range(len(data[0])):
            if not np.isnan(data_threshold[i, j]):
                text = ax.text(j, i, "{:.2f}".format(data[i, j]),
                            ha="center", va="center", color="w", fontsize=6)

    # set the tick labels
    ax.set_xticks(np.arange(len(data[0])))
    ax.set_yticks(np.arange(len(data)))
    ax.set_xticklabels(hparams['input_labels'], rotation = 90)
    ax.set_yticklabels(hparams['input_labels'])

    # set the colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # set the title and axis labels
    ax.set_title("Pearson Corr Coeff of Inputs (threshold={:.1f})".format(threshold))
    # ax.set_xlabel("X-axis label")
    # ax.set_ylabel("Y-axis label")


    # # show the plot
    # plt.show()


def get_dir_of_saving_file(hparams, type_save):
    """type_save can be from ['data-aspects', ...] """
    
    hparams['save_dir']
    hparams['experiment']
    hparams['mouse']
    hparams['sessions']
    hparams['analysis_experiment_name']

    # get the numpy ndarray of sessions
    sess_array = hparams['sessions']
    # Convert the ndarray to a list of string elements
    sess_list_of_str = [str(x) for x in sess_array.tolist()]
    # Convert the array to a string and use it as the name of a session-directory
    sess_dir_name = os.path.join('-'.join(sess_list_of_str))

    if type_save == 'data-aspects':
        subdirectory = 'data-aspects-dir'
        path_to_save = os.path.join(hparams['save_dir'], hparams['experiment'], hparams['mouse'], \
                                    sess_dir_name, hparams['analysis_experiment_name'], subdirectory )

    if type_save == 'sth-related-to-num-of-latent-states':
        pass
        # to be completed

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
        print(f"Subdirectory '{subdirectory}' created successfully for the file to be saved in.")
    else:
        print(f"Subdirectory '{subdirectory}' already exists for the file to be saved in.")

    return path_to_save


def save_plot(hparams, type_save, plot_name):

    path_to_save = get_dir_of_saving_file(hparams, type_save)
    # Save the plot as a PNG file
    plt.savefig(os.path.join(path_to_save , plot_name), dpi=300, bbox_inches='tight')


def save_file(hparams, type_save, file_name, file_to_save):
    import pickle
    path_to_save = get_dir_of_saving_file(hparams, type_save)
    # Open the file in write mode
    file_path = os.path.join(path_to_save, file_name)
    with open(file_path, "wb") as file1:
            # Write the data to the file
        pickle.dump(file_to_save, file1)



# def plot_choice_vs_trials(hparams, choice_list):
#     tot_trials = hparams['num_trials_in_input_array']
#     n_trials_in_sessions_minus_history_step = np.array(hparams['list_of_trial_count_in_chosen_sessens']) - hparams['num_history_step']
#     cumulative_trila_num = np.cumsum(n_trials_in_sessions_minus_history_step)

#     fig, ax = plt.subplots(figsize=(20, 3), dpi=80, facecolor='w', edgecolor='k')
#     plt.step(range(tot_trials),choice_list[0][range(tot_trials)], color = "red")

#     plt.axvline(x=0, color='black', linestyle='--', label='session devider')
#     for num_trials in cumulative_trila_num:
#         plt.axvline(x=num_trials, color='black', linestyle='--')
        
#     plt.yticks([0, 1, 2], get_output_class_list(hparams))
#     plt.title(hparams['mouse'] + " Choice - sessions " + str(hparams['sessions']))
#     plt.xlabel("trial #", fontsize = 15)
#     plt.ylabel("observation class", fontsize = 15)
#     plt.legend()

def plot_choice_vs_trials(hparams, choice_list):
    tot_trials = hparams['num_trials_in_input_array']
    n_trials_in_sessions_minus_history_step = np.array(hparams['list_of_trial_count_in_chosen_sessens']) - hparams['num_history_step']
    cumulative_trila_num = np.cumsum(n_trials_in_sessions_minus_history_step)
    choice_array = choice_list[0].copy()
    
    if hparams['output_form'] == 'R0_nan1_L2':
        choice_array[choice_array == 2] = 3
        choice_array[choice_array == 1] = 2
        choice_array[choice_array == 3] = 1

    print(choice_array[100:200,0])


    fig, ax = plt.subplots(figsize=(20, 3), dpi=80, facecolor='w', edgecolor='k')
    plt.step(range(tot_trials),choice_array[range(tot_trials)], color = "red")

    plt.axvline(x=0, color='black', linestyle='--', label='session devider')
    for num_trials in cumulative_trila_num:
        plt.axvline(x=num_trials, color='black', linestyle='--')
        
    plt.yticks([0, 1, 2], ['Right' , 'Left', 'Ignored'])
    plt.title(hparams['mouse'] + " Choice - sessions " + str(hparams['sessions']))
    plt.xlabel("trial #", fontsize = 15)
    plt.ylabel("observation class", fontsize = 15)
    plt.legend()


def plot_log_prob_of_model_vs_EM_iteration_output_of_fit_fun_training_progress(fit_ll):
    """
    Plot the log probabilities of fit model as a function of EM_iteration.
    This plot shows the training progress, and is the output of fit function ( glmhmm.fit() ).
    ? Fit model final LL should be greater than or equal to true LL.
    """
    fig = plt.figure(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(fit_ll, label="EM")
    plt.legend(loc="lower right")
    plt.xlabel("EM Iteration")
    plt.xlim(0, len(fit_ll))
    plt.ylabel("Log Probability")
    plt.show()



def plot_weights_per_class_and_transition_matrix(hparams, weights, log_transitions, num_states):
    """This function thakes the set of learned_weights and tuple of log_transitions, and plot them (weights are plotted for each 
        output class)
    
    inputs:
        weights: numpy.ndarray - size : (num_states, num_categories - 1, input_dim )
        log_transitions: tuple - size : (1, num_states, num_states)  [log_transitions[0]: numpy.ndarray - size : (num_states, num_states)]
        num_states : scaler
        """
    fig = plt.figure(figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
    plt.subplots_adjust(wspace=0.3, hspace=0.6)
    # plt.subplot(1, 2, 1)
    cols = hparams['colors']
    class_list = get_output_class_list(hparams)
    num_categories = hparams['num_categories']
    input_dim = hparams['input_dim']

    # k=1
    # plt.plot(range(input_dim), weights[k,c], marker='o', linestyle = '-',
    #     color=cols[k], lw=1.5, label="state " + str(k+1) + "; class " + str(c+1))
    
    for c in range(1):#(num_categories):
        plt.subplot(num_categories+1, 1, c + 1)
        if c < num_categories-1:
            for k in range(num_states):
                plt.plot(range(input_dim), weights[k,c], marker='o', linestyle = '-',
                    color=cols[k], lw=1.5, label="state " + str(k+1) + "; class " + str(c+1))
        else:
            for k in range(num_states):
                plt.plot(range(input_dim), np.zeros(input_dim), marker='o', linestyle = '--',
                        color=cols[k], lw=1.5, label="state " + str(k+1) + "; class " + str(c+1), alpha = 0.5)

        plt.axhline(y=0, color="k", alpha=0.5, ls="--")
        plt.yticks(fontsize=10)
        # plt.xlabel("covariate", fontsize=15)
        # if c == 0:
        plt.ylabel("GLM weight", fontsize=15)
        plt.xticks(range(input_dim), hparams['input_labels'], fontsize=12, rotation=90)
        plt.legend()
        plt.title("Weights; choice class : " + class_list[c], fontsize = 15)
        plt.ylim((-12,28))
        
    plt.subplot(num_categories+1, 1, num_categories+1)
    learned_trans_mat = np.exp(log_transitions)[0]
    plt.imshow(learned_trans_mat, vmin=-0.8, vmax=1, cmap='bone')
    for i in range(learned_trans_mat.shape[0]):
        for j in range(learned_trans_mat.shape[1]):
            text = plt.text(j, i, str(np.around(learned_trans_mat[i, j], decimals=2)), ha="center", va="center",
                            color="k", fontsize=12)
    plt.xlim(-0.5, num_states - 0.5)
    plt.xticks(range(0, num_states), [str(i) for i in range(1, num_states+1)], fontsize=10)
    plt.yticks(range(0, num_states), [str(i) for i in range(1, num_states+1)], fontsize=10)
    plt.ylim(num_states - 0.5, -0.5)
    plt.ylabel("state t", fontsize = 15)
    plt.xlabel("state t+1", fontsize = 15)
    plt.title("Transition matrix", fontsize = 15)



def plot_LL_for_model_selection(test_ll, train_ll, test_ll_base, train_ll_base, num_states_2cv, kw):
    from scipy.stats import sem

    n_repeats = np.array(test_ll).shape[1]
    fig, ax = plt.subplots()
    cols = [["red","black"],["orange","blue"]]
    labels = [["test","train"],["test_base","train_base"]]

    if kw == "both_data":
        test_set = [test_ll, test_ll_base]
        train_set = [train_ll, train_ll_base]

        test_ll_set_ = [np.array(test_ll), np.array(test_ll_base)]
        train_ll_set_ = [np.array(train_ll), np.array(train_ll_base)]


        for i in range(2):
            # Calculate the mean of each array in the two lists
            means_test_ll = [np.mean(arr) for arr in test_set[i]]
            sem_test_ll = [np.std(arr) for arr in test_set[i]]
            sem_test_ll = np.nan_to_num(sem_test_ll, nan=0)

            means_train_ll = [np.mean(arr) for arr in train_set[i]]
            sem_train_ll = [np.std(arr) for arr in train_set[i]]
            sem_train_ll = np.nan_to_num(sem_train_ll, nan=0)

            for j in num_states_2cv:
                ax.plot(np.ones((n_repeats,))*(j), test_ll_set_[i][j-1,:],'o', color=cols[i][0] , alpha=.25)
                ax.plot(np.ones((n_repeats,))*(j), train_ll_set_[i][j-1,:],'o', color=cols[i][1] , alpha=.25)

            # Plot the means and standard errors of the means of the arrays in the first list in order
            ax.errorbar([num + .15 for num in num_states_2cv], means_test_ll, yerr=sem_test_ll, label=labels[i][0], fmt="o", color=cols[i][0])

            # Plot the means and standard errors of the means of the arrays in the first list in order
            ax.errorbar([num + .15 for num in num_states_2cv], means_train_ll, yerr=sem_train_ll, label=labels[i][1], fmt="o", color=cols[i][1])

    
    else:
        test_ll_ = np.array(test_ll)
        train_ll_ = np.array(train_ll)
        # Calculate the mean of each array in the two lists
        means_test_ll = [np.mean(arr) for arr in test_ll]
        sem_test_ll = [np.std(arr) for arr in test_ll]
        sem_test_ll = np.nan_to_num(sem_test_ll, nan=0)

        means_train_ll = [np.mean(arr) for arr in train_ll]
        sem_train_ll = [np.std(arr) for arr in train_ll]
        sem_train_ll = np.nan_to_num(sem_train_ll, nan=0)

        for i in num_states_2cv:
            ax.plot(np.ones((n_repeats,))*(i), test_ll_[i-1,:],'or', alpha=.25)
            ax.plot(np.ones((n_repeats,))*(i), train_ll_[i-1,:],'ok', alpha=.25)

        # Plot the means and standard errors of the means of the arrays in the first list in order
        ax.errorbar([num + .15 for num in num_states_2cv], means_test_ll, yerr=sem_test_ll, label="Test", fmt="or")

        # Plot the means and standard errors of the means of the arrays in the first list in order
        ax.errorbar([num + .15 for num in num_states_2cv], means_train_ll, yerr=sem_train_ll, label="Train", fmt="ok")

    plt.xlabel("#latent")
    plt.ylabel("LL/trial")
    plt.title("Model selection - cross validation")
    plt.xticks(num_states_2cv)

    plt.legend()
    plt.show()




def plot_Relative_LL_for_model_selection(test_ll, train_ll, test_ll_base, train_ll_base, num_states_2cv, kw):
    from scipy.stats import sem

    n_repeats = np.array(test_ll).shape[1]
    fig, ax = plt.subplots()
    # cols = [["red","black"],["orange","blue"]]
    # labels = [["test","train"],["test_base","train_base"]]

    test_ll_ = np.array(test_ll)
    train_ll_ = np.array(train_ll)

    # Calculate the mean of each array in the two lists

    # if kw == "both_data":
    # test_set = [test_ll, test_ll_base]
    # train_set = [train_ll, train_ll_base]

    # test_ll_set_ = [np.array(test_ll), np.array(test_ll_base)]
    # train_ll_set_ = [np.array(train_ll), np.array(train_ll_base)]

    # means_test_ll = np.sum(test_ll_,axis=1)

    # for i in range(2):
        # Calculate the mean of each array in the two lists
    means_test_ll = [np.mean(arr) for arr in test_ll]
    sem_test_ll = [np.std(arr) for arr in test_ll]
    sem_test_ll = np.nan_to_num(sem_test_ll, nan=0)

    means_train_ll = [np.mean(arr) for arr in train_ll]
    sem_train_ll = [np.std(arr) for arr in train_ll]
    sem_train_ll = np.nan_to_num(sem_train_ll, nan=0)

    means_test_ll_base = [np.mean(arr) for arr in test_ll_base]
    means_train_ll_base = [np.mean(arr) for arr in train_ll_base]

    for j in num_states_2cv:
        ax.plot(np.ones((n_repeats,))*(j), test_ll_[j-1,:] - np.mean(means_test_ll_base),'or' , alpha=.25)
        ax.plot(np.ones((n_repeats,))*(j), train_ll_[j-1,:] - np.mean(means_train_ll_base),'ok', alpha=.25)

    # Plot the means and standard errors of the means of the arrays in the first list in order
    ax.errorbar([num + .15 for num in num_states_2cv], means_test_ll-np.mean(np.array(means_test_ll_base)), yerr=sem_test_ll, label="Test", fmt="or")

    # Plot the means and standard errors of the means of the arrays in the first list in order
    ax.errorbar([num + .15 for num in num_states_2cv], means_train_ll-np.mean(np.array(means_train_ll_base)), yerr=sem_train_ll, label='Train', fmt="ok")

    
    # else:
    #     test_ll_ = np.array(test_ll)
    #     train_ll_ = np.array(train_ll)
    #     # Calculate the mean of each array in the two lists
    #     means_test_ll = [np.mean(arr) for arr in test_ll]
    #     sem_test_ll = [sem(arr) for arr in test_ll]
    #     sem_test_ll = np.nan_to_num(sem_test_ll, nan=0)

    #     means_train_ll = [np.mean(arr) for arr in train_ll]
    #     sem_train_ll = [sem(arr) for arr in train_ll]
    #     sem_train_ll = np.nan_to_num(sem_train_ll, nan=0)

    #     for i in num_states_2cv:
    #         ax.plot(np.ones((n_repeats,))*(i), test_ll_[i-1,:]-np.mean(means_test_ll),'or', alpha=.25)
    #         ax.plot(np.ones((n_repeats,))*(i), train_ll_[i-1,:]-np.mean(means_test_ll),'ok', alpha=.25)

    #     # Plot the means and standard errors of the means of the arrays in the first list in order
    #     ax.errorbar([num + .15 for num in num_states_2cv], means_test_ll, yerr=sem_test_ll, label="Test", fmt="or")

    #     # Plot the means and standard errors of the means of the arrays in the first list in order
    #     ax.errorbar([num + .15 for num in num_states_2cv], means_train_ll, yerr=sem_train_ll, label="Train", fmt="ok")

    plt.xlabel("#latent")
    plt.ylabel("Relative LL/trial")
    plt.title("Model selection - cross validation")
    plt.xticks(num_states_2cv)

    plt.legend()
    plt.show()






def plot_evolution_of_frac_occ(hparams, frac_occupancies_per_sess):
    # Create subplots with a 3x1 grid
    num_states = frac_occupancies_per_sess.shape[0]
    colors = hparams['colors']

    fig, axes = plt.subplots(nrows=num_states, ncols=1, figsize=(6, 8))

    # Iterate over each row and plot a bar plot in each subplot
    for i, (row, color) in enumerate(zip(frac_occupancies_per_sess, colors[:num_states])):
        axes[i].bar(range(len(row)), row, color= color)
        axes[i].set_xlabel('# Session')
        axes[i].set_ylabel('frac occ')
        axes[i].set_ylim([-0.01,1.01])
        axes[i].set_title('State {}'.format(i+1))

    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.show()