import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import cm
import numpy as np


# scale and move the coordinates -> fit [0, 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def sne_visual(test_embeddings, test_predictions):
    # test_embeddings = test_embeddings.cpu().numpy()
    # test_predictions = test_predictions.cpu().numpy()
    print('Dimensions of latent feature before t-SNE:', test_embeddings.shape)

    # use tSNE to reduce dimension
    tsne = TSNE(n_components=2, init="pca", random_state=0, learning_rate=500.)
    tsne_proj = tsne.fit_transform(test_embeddings)
    print('Dimensions after tSNE-2D:', tsne_proj.shape)

    x_min, x_max = np.min(tsne_proj, 0), np.max(tsne_proj, 0)
    tsne_proj = (tsne_proj - x_min) / (x_max - x_min)  # 对数据进行归一化处理

    # tx = scale_to_01_range(tx)
    # ty = scale_to_01_range(ty)

    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8, 8))
    num_categories = 16
    for label in range(num_categories):
        indices = label == test_predictions
        ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1],
                   c=np.array(cmap(label)).reshape(1, 4), label=label)
    # ax.legend(fontsize='small', markerscale=2)
    # plt.show()
    ax.axis("off")
    # finally, save the plot
    plt.savefig('./tsne_z_so3.jpg', bbox_inches='tight', pad_inches=0)

    # # extract x and y coordinates representing the positions of the images on T-SNE plot
    # tx = latent_feat_tsne_2d[:, 0]
    # ty = latent_feat_tsne_2d[:, 1]
    #
    # tx = scale_to_01_range(tx)
    # ty = scale_to_01_range(ty)
    #
    # # initialize a matplotlib plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # colors_per_class = [i for i in range(40)]
    #
    # # for every class, we'll add a scatter plot separately
    # for label in colors_per_class:
    #     # find the samples of the current class in the data
    #     indices = [i for i, l in enumerate(labels) if l == label]
    #
    #     # extract the coordinates of the points of this class only
    #     current_tx = np.take(tx, indices)
    #     current_ty = np.take(ty, indices)
    #
    #     # convert the class color to matplotlib format
    #     color = np.array(colors_per_class[label], dtype=np.float) / 255
    #
    #     # add a scatter plot with the corresponding color and label
    #     ax.scatter(current_tx, current_ty, c=color, label=label)

    # # plot the points projected with PCA and tSNE
    # fig = plt.figure()
    # fig.suptitle('t-SNE Visualization')
    # ax = fig.add_subplot(111)
    # ax.title.set_text('tSNE')
    # ax.axis("off")
    # ax.scatter(latent_feat_tsne_2d[:, 0], latent_feat_tsne_2d[:, 1], c=label, s=30, cmap='Set1')
    # plt.savefig('./tsne.jpg', bbox_inches='tight', pad_inches=0)
