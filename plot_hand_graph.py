import pathlib
import random
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd


test_path = 'plots/test2'


def save_x_to_csv(x_initial_df, x_preprocessed_df, x_conv_df):
    # limit the number of files saved
    if random.randint(0, 100) < 1:
        # Save x
        file_name = 'x_batch{}_{:%H%M%S}.csv'.format(0, datetime.now())
        # create the directory if does not exist, without raising an error if it does already exist
        pathlib.Path('plots/test2').mkdir(parents=True, exist_ok=True)
        # save the file
        x_df = pd.concat([x_initial_df, x_preprocessed_df, x_conv_df], axis=1)
        x_df.to_csv(path_or_buf='{}/{}'.format(test_path, file_name))


def update(t, lines, skeletons_display, ax, seq_length):
    ax.set_title('Timestep : {}/{}'.format(t, seq_length), loc='right')

    for b, line in enumerate(lines):
        # set x and y data
        line.set_data(skeletons_display[t, 0:2, :, b])
        # set z data
        line.set_3d_properties(skeletons_display[t, 2, :, b])
    return lines


def main(file_name, preprocessed):
    # Idx of the bones in the hand skeleton to display it.

    bones = np.array([
        [0, 1],
        [0, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [1, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [1, 10],
        [10, 11],
        [11, 12],
        [12, 13],
        [1, 14],
        [14, 15],
        [15, 16],
        [16, 17],
        [1, 18],
        [18, 19],
        [19, 20],
        [20, 21]
    ])

    image_name = file_name
    image_path = '{}/{}'.format(test_path, image_name)

    # ---------
    # Get graph data
    # ---------

    # pandas DataFrame, 100 rows, n*66 columns
    df = pd.read_csv(image_path).iloc[:, 1:]

    # Initial skeleton : [:, 0:66], preprocessed : [:, 66:132]
    if preprocessed:
        skeleton = df.iloc[:, 66:132].values
    else:
        skeleton = df.iloc[:, 0:66].values

    nb_bones = bones.shape[0]
    seq_length = skeleton.shape[0]

    # (x, y, z) coordinates of bones (for each bone there are 2 joints)
    # x : (vertex = 1 or 2, x_v)
    x = np.zeros([2, bones.shape[0]])
    y = np.zeros([2, bones.shape[0]])
    z = np.zeros([2, bones.shape[0]])

    # skeletons_display = (t, axis, vertex, bone)
    skeletons_display = np.zeros([seq_length, 3, 2, bones.shape[0]])

    for t in range(0, seq_length):
        for idx_bones in range(0, bones.shape[0]):
            joint1 = bones[idx_bones, 0]
            joint2 = bones[idx_bones, 1]

            # (x, y, z) coordinates of joint1 and joint2
            pt1 = skeleton[t, joint1 * 3:joint1 * 3 + 3]
            pt2 = skeleton[t, joint2 * 3:joint2 * 3 + 3]

            x[0, idx_bones] = pt1[0]
            x[1, idx_bones] = pt2[0]
            y[0, idx_bones] = pt1[1]
            y[1, idx_bones] = pt2[1]
            z[0, idx_bones] = pt1[2]
            z[1, idx_bones] = pt2[2]

        skeletons_display[t, 0, :, :] = x
        skeletons_display[t, 1, :, :] = y
        skeletons_display[t, 2, :, :] = z

    # ---------
    # Plot graph
    # ---------

    # Create plot
    fig = plt.figure()

    # Initial skeleton
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    lines = [ax.plot(skeletons_display[0, 0, :, b], skeletons_display[0, 1, :, b], skeletons_display[0, 2, :, b])[0]
             for b in range(nb_bones)]
    line_ani = animation.FuncAnimation(fig, update, frames=seq_length, fargs=(lines, skeletons_display, ax, seq_length),
                                       interval=50,
                                       blit=False)

    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='plot hand graph')
    parser.add_argument('--file', metavar='file', required=True,
                        help='File name of the .csv')
    parser.add_argument('--preprocessed', metavar='preprocessed', required=True,
                        help='File name of the .csv')
    args = parser.parse_args()

    main(file_name=args.file, preprocessed=args.preprocessed)
