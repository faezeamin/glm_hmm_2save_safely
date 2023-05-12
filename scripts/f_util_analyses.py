import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from f_util_preprocess import *


def get_pearson_corr_coeff_of_input_data(input_design_matrix_list):
    a = np.array(input_design_matrix_list).squeeze()
    corr_ = np.corrcoef(a.T)
    return corr_

def decompose_correlated_data_using_PCA(dataset):
    """
    This function takes the correlated data and de-correated it using PCA. the output has the same dimension as the input.
    input: data: numpy ndarray , size(sample_size , n_features). data is correlated_data
    output: transformed_data numpy ndarray , size(sample_size , n_features). decorreated data 
    """
    from sklearn.decomposition import PCA

    # Perform PCA on the data set
    pca = PCA(n_components=np.shape(dataset)[1])
    pca.fit(dataset)
    transformed_data = pca.transform(dataset)

    #get loadings of principal componets
    loadings = pca.components_

    # print(np.cov(transformed_data, rowvar=False))
    # im = plt.imshow(np.cov(transformed_data, rowvar=False), cmap="seismic") #, vmin=-1, vmax=1)
    
    return transformed_data, loadings, pca

def decompose_input_list(input_design_matrix_list):
    """
    This function takes the input_list, decomposes it using PCA, and returns a de-correlated list of input_data with the same size as the input.

    input: input_design_matrix_list : list - size(1 , sample_size , n_features) 
    output: decorr_input_list: list - size(1 , sample_size , n_features)
     """
    dataset = np.array(input_design_matrix_list).squeeze()
    decorr_input_list = []

    # Perform PCA on the data set
    transformed_data, loadings, pca = decompose_correlated_data_using_PCA(dataset)
    decorr_input_list.append(transformed_data)
    
    return decorr_input_list, loadings, pca


def cv(hparams, num_states_2cv, n_repeats, heldout_frac, inpts, true_choices, kw):
    """kw: choose from ["true_data", "sythetic_data, "both_data]"""
    from ssm.model_selection import cross_val_scores
    import ssm

    if kw == "true_data":
        test_ll, train_ll = [] , []
        for num_sts in num_states_2cv:
            mle_glmhmm_cv = ssm.HMM(num_sts, hparams['observation_dim'], hparams['input_dim']   , observations="input_driven_obs", 
                    observation_kwargs=dict(C=hparams['num_categories']), transitions="standard")
        
            print(num_sts)

            test_scores, train_scores = cross_val_scores(
                mle_glmhmm_cv, true_choices, inputs=inpts, masks=None, tags=None,
                heldout_frac=heldout_frac, n_repeats=n_repeats)
            
            test_ll.append(test_scores.copy())
            train_ll.append(train_scores.copy())
            

            print('test', test_scores)
            print('train',train_scores)
        
        return test_ll, train_ll


    if kw == "sythetic_data":
        fake_choice_base = generate_synthetic_choice(hparams, true_choices)

        test_ll_base, train_ll_base = [] , []
        for num_sts in num_states_2cv:
            mle_glmhmm_cv = ssm.HMM(num_sts, hparams['observation_dim'], hparams['input_dim']   , observations="input_driven_obs", 
                    observation_kwargs=dict(C=hparams['num_categories']), transitions="standard")
        
            print(num_sts)

            test_scores, train_scores = cross_val_scores(
                mle_glmhmm_cv, fake_choice_base, inputs=inpts, masks=None, tags=None,
                heldout_frac=heldout_frac, n_repeats=n_repeats)
            
            test_ll_base.append(test_scores.copy())
            train_ll_base.append(train_scores.copy())
            

            print('test', test_scores)
            print('train',train_scores)
        
        return test_ll_base, train_ll_base
    
    if kw == "both_data":
        fake_choice_base = generate_synthetic_choice(hparams, true_choices)

        test_ll, train_ll , test_ll_base, train_ll_base = [] , [], [] , []

        for num_sts in num_states_2cv:
            mle_glmhmm_cv = ssm.HMM(num_sts, hparams['observation_dim'], hparams['input_dim']   , observations="input_driven_obs", 
                    observation_kwargs=dict(C=hparams['num_categories']), transitions="standard")
        
            print(num_sts)

            # do cv on real data
            test_scores, train_scores = cross_val_scores(
                mle_glmhmm_cv, true_choices, inputs=inpts, masks=None, tags=None,
                heldout_frac=heldout_frac, n_repeats=n_repeats)
            
            test_ll.append(test_scores.copy())
            train_ll.append(train_scores.copy())
            
            print('true data')
            print('test_true', test_scores)
            print('train_true',train_scores)

            # do cv on synthetic data
            test_scores, train_scores = cross_val_scores(
                mle_glmhmm_cv, fake_choice_base, inputs=inpts, masks=None, tags=None,
                heldout_frac=heldout_frac, n_repeats=n_repeats)
            
            test_ll_base.append(test_scores.copy())
            train_ll_base.append(train_scores.copy())
            
            print('synthetic data')
            print('test_synth', test_scores)
            print('train_synth',train_scores)
        
        return test_ll, train_ll, test_ll_base, train_ll_base





def do_t_SNE(inpts_arr, choices_arr, n_components):
    """Input: 
        inpts_arr: nd Array size(n_data_points, n_fetures)
        choices_arr: 1d Array size(n_data_points, 1)
    """
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from sklearn.datasets import make_blobs
    from sklearn.manifold import TSNE

    # Generate random dataset
    # X, y = make_blobs(n_samples=500, n_features=4, centers=3, random_state=42)
    X, y = inpts_arr, choices_arr

    # Perform t-SNE
    tsne = TSNE(n_components = n_components, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Plot the t-SNE results
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
    plt.title("t-SNE Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()



def do_pca(data_4pca):

    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D
    # %matplotlib notebook
    # %matplotlib auto
    # # Generate or load the data array
    # data = np.random.rand(100, 5)  # Replace with your own data
    
    n_components = 5
    # Perform PCA
    if data_4pca.shape[1]<5:
        n_components = data_4pca.shape[1]

    pca = PCA(n_components=n_components)
    # pca = PCA()
    pca_data = pca.fit_transform(data_4pca)

    # Extract the PC1 and PC2 dimensions
    pc1 = pca_data[:, 0]
    pc2 = pca_data[:, 1]
    pc3 = pca_data[:, 3]


    # Create a 3D plot for PC1, PC2, and PC3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc1, pc2, pc3)

    # Set labels for the axes
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    # Show the plot
    plt.show()
    # plot_name = 'uu'
    # save_plot(hparams, type_save= 'data-aspects', plot_name = plot_name)

    #  Plot the data in PC1 and PC2 dimensions
    plt.scatter(pc1, pc2)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Data in PC1 and PC2 Dimensions')
    plt.show()

    #  Plot the data in PC1 and PC2 dimensions
    plt.scatter(pc1, pc3)
    plt.xlabel('PC1')
    plt.ylabel('PC3')
    plt.title('Data in PC1 and PC3 Dimensions')
    plt.show()

        #  Plot the data in PC1 and PC2 dimensions
    plt.scatter(pc2, pc3)
    plt.xlabel('PC2')
    plt.ylabel('PC3')
    plt.title('Data in PC2 and PC3 Dimensions')
    plt.show()

    # Get the variance explained by each principal component
    variance_explained = pca.explained_variance_ratio_

    # Print the variance explained by each principal component
    for i, variance in enumerate(variance_explained):
        print(f"PC{i+1} Variance Explained: {variance:.2%}")