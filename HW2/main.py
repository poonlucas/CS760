import math
import random
import sys
import numpy as np
import graphviz
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.tree import DecisionTreeClassifier
from scipy.interpolate import lagrange


class Node:
    def __init__(self):
        self.value = None
        self.threshold = None
        self.feature = None
        self.left = None
        self.right = None
        self.gain = None


def entropy(pos, neg):
    if pos == 0 or neg == 0:
        return 0
    sum = pos + neg
    return -(pos / sum) * math.log((pos / sum), 2) - (neg / sum) * math.log((neg / sum), 2)


def findBestSplit(data, c, out=False):
    total_entropy = entropy(np.count_nonzero(data[:, -1] == 1), np.count_nonzero(data[:, -1] == 0))
    max_gainRatio = -np.inf
    max_threshold = -np.inf
    max_i = 0
    for i, candidates in enumerate(c):
        for threshold in candidates:
            # [pos, neg]
            left = [0, 0]
            right = [0, 0]
            for idx, datapoint in enumerate(data[:, i]):
                if datapoint >= threshold:  # left tree
                    if data[idx, -1] == 1:
                        left[0] += 1
                    elif data[idx, -1] == 0:
                        left[1] += 1
                else:
                    if data[idx, -1] == 1:
                        right[0] += 1
                    elif data[idx, -1] == 0:
                        right[1] += 1
            left_entropy = entropy(left[0], left[1])
            right_entropy = entropy(right[0], right[1])
            total_left = left[0] + left[1]
            total_right = right[0] + right[1]
            infoGain = total_entropy - ((total_left / data[:, 0].size) * left_entropy +
                                        (total_right / data[:, 0].size) * right_entropy)

            if infoGain == 0:
                if out:
                    print("x_{} >= {} & {}".format(i + 1, threshold, 0.0))
                continue

            left_splitInfo = 0
            right_splitInfo = 0
            if total_left != 0:
                left_splitInfo = -(total_left / data[:, 0].size) * math.log((total_left / data[:, 0].size), 2)
            if total_right != 0:
                right_splitInfo = -(total_right / data[:, 0].size) * math.log((total_right / data[:, 0].size), 2)
            splitInfo = left_splitInfo + right_splitInfo
            gainRatio = infoGain / splitInfo

            if out:
                print("x_{} >= {} & {}".format(i + 1, threshold, gainRatio))

            if max_gainRatio < gainRatio:
                max_gainRatio = gainRatio
                max_threshold = threshold
                max_i = i
    return max_gainRatio, max_threshold, max_i


def makeSubtree(data, out=False):
    # determine candidate splits
    c = []
    # Stopping Criteria 1: Node is empty
    if data.size == 0:
        node = Node()
        node.value = 1
        return node
    else:
        for i in range(data[0].size - 1):
            c.append(np.unique(data[:, i]))

        node = Node()
        # Find best split
        s = findBestSplit(data, c, out)

        if s[0] <= 0:
            pos = np.count_nonzero(data[:, -1] == 1)
            neg = np.count_nonzero(data[:, -1] == 0)
            if pos >= neg:
                node.value = "Prediction: 1"
            else:
                node.value = "Prediction: 0"
        else:
            if out:
                print("x_{} >= {}, Gain Ratio: {}".format(s[2], s[1], s[0]))  # print split
            node.feature = s[2] + 1
            node.threshold = s[1]
            node.gain = s[0]
            node.value = "x_{} >= {}".format(node.feature, node.threshold)
            data1 = data[data[:, s[2]] >= s[1]]
            data2 = data[data[:, s[2]] < s[1]]
            if out:
                print("\nLeft Subtree")
            node.left = makeSubtree(data1)
            if out:
                print("\nRight Subtree")
            node.right = makeSubtree(data2)

        return node


def create_tree(node, dot, num=0, gr=True):
    num += 1
    curr_num = num
    if gr and (node.left is not None and node.right is not None):
        dot.node(str(num), "{}, Gain Ratio: {}".format(node.value, node.gain))
    else:
        dot.node(str(num), node.value)
    if node.left is not None:
        dot.edge(str(curr_num), str(num + 1))
        num = create_tree(node.left, dot, num, gr=gr)

    if node.right is not None:
        dot.edge(str(curr_num), str(num + 1))
        num = create_tree(node.right, dot, num, gr=gr)

    return num


def rectangles(node, max_x1, max_x2, min_x1, min_x2, parent=None):
    rectangle_list = []

    threshold = node.threshold
    feature = node.feature

    if node.left is None and node.right is None:
        rect = Rectangle((0, 0), 0, 0)
        if int(node.value[-1]) == 1:
            rect = Rectangle((min_x1, min_x2), max_x1 - min_x1, max_x2 - min_x2, facecolor=("red", 0.5))
        elif int(node.value[-1]) == 0:
            rect = Rectangle((min_x1, min_x2), max_x1 - min_x1, max_x2 - min_x2, facecolor=("blue", 0.5))
        rectangle_list.append(rect)
        return rectangle_list

    if node.left is not None:
        new_min_x1 = min_x1
        new_min_x2 = min_x2
        if feature == 1:
            new_min_x1 = max(min_x1, threshold)
        elif feature == 2:
            new_min_x2 = max(min_x2, threshold)
        rectangle_list.extend(rectangles(node.left, max_x1, max_x2, new_min_x1, new_min_x2, parent=node))
    if node.right is not None:
        new_max_x1 = max_x1
        new_max_x2 = max_x2
        if feature == 1:
            new_max_x1 = min(max_x1, threshold)
        elif feature == 2:
            new_max_x2 = min(max_x2, threshold)
        rectangle_list.extend(rectangles(node.right, new_max_x1, new_max_x2, min_x1, min_x2, parent=node))

    return rectangle_list


def predict(data_point, node):
    if node.left is None and node.right is None:
        return node.value[-1]

    if data_point[node.feature - 1] >= node.threshold:
        prediction = predict(data_point, node.left)
    else:
        prediction = predict(data_point, node.right)

    return prediction


def count(node):
    if node is None:
        return 0
    num_nodes = 1
    if node.left is not None:
        num_nodes += count(node.left)
    if node.right is not None:
        num_nodes += count(node.right)
    return num_nodes


def eval_tree(data, tree):
    err = 0
    num_nodes = count(tree)
    test_size = data.size / 3
    for data_point in data:
        prediction = predict(data_point, tree)
        if int(data_point[-1]) != int(prediction):
            err += 1
    return err / test_size, num_nodes


if __name__ == '__main__':

    ##################################################################################################

    # For Q1-3

    # default split
    training_pa = 1

    # Read user input
    if len(sys.argv) < 2:
        print("Invalid number of args")
        exit()
    data = np.loadtxt(sys.argv[1])

    # Shuffle data
    np.random.shuffle(data)

    if len(sys.argv) > 2:
        training_pa = float(sys.argv[2])

    training_size = data[:, 0].size * training_pa

    training_data = data[:int(training_size), :]
    test_data = data[int(training_size):, :]

    print()

    ##################################################################################################

    # For < Q2.7
    # tree = makeSubtree(training_data, out=True)  # makeSubtree(data, out=True) for candidate splits
    # print()
    #
    # # Tree Visualization
    # dot = graphviz.Digraph()
    # create_tree(tree, dot, gr=False)  # gr=True for Gain Ratio in tree
    # # print(dot.source)  # print nodes and edges
    # dot.view()
    #
    # # Scatter Plot
    # fig, ax = plt.subplots()
    # ax.scatter(training_data[:, 0], training_data[:, 1], c=training_data[:, -1], s=10, cmap='bwr')
    # plt.xlabel("x_1")
    # plt.ylabel("x_2")
    # plt.title("D1 Scatter + Decision Boundary (red: Y=1, blue: Y=0)")
    # rectangle_list = rectangles(tree, max_x1=training_data[:, 0].max(), max_x2=training_data[:, 1].max(),
    #                             min_x1=training_data[:, 0].min(), min_x2=training_data[:, 1].min())
    # for rect in rectangle_list:
    #     ax.add_patch(rect)
    # plt.show()

    ##################################################################################################

    # For Q2.7 simulate learning curve
    error = []
    num_nodes = []
    n = [32, 128, 512, 2048, 8192]

    for i in range(5, 15, 2):
        training_data_subset = training_data[:int(math.pow(2, i)), :]
        tree = makeSubtree(training_data_subset, out=False)
        print()

        # Eval
        err, nodes_count = eval_tree(test_data, tree)
        error.append(err)
        num_nodes.append(nodes_count)

        # Tree Visualization
        dot = graphviz.Digraph()
        create_tree(tree, dot, gr=False)
        # print(dot.source)  # print nodes and edges
        dot.view()

        # Scatter Plot
        fig, ax = plt.subplots()
        ax.scatter(training_data_subset[:, 0], training_data_subset[:, 1], c=training_data_subset[:, -1], s=10,
                   cmap='bwr')
        plt.xlabel("x_1")
        plt.ylabel("x_2")
        plt.title("D_{} (red: Y=1, blue: Y=0)".format(int(math.pow(2, i))))
        rectangle_list = rectangles(tree, max_x1=training_data[:, 0].max(), max_x2=training_data[:, 1].max(),
                                    min_x1=training_data[:, 0].min(), min_x2=training_data[:, 1].min())
        for rect in rectangle_list:
            ax.add_patch(rect)
        plt.show()

    for idx, err in enumerate(error):
        print(n[idx], num_nodes[idx], error[idx])
    plt.plot(n, error)
    plt.xlabel("n")
    plt.ylabel("Error")
    plt.title("n vs Error")
    plt.show()

    ##################################################################################################

    # For Q3 sklearn
    # error = []
    # num_nodes = []
    # n = [32, 128, 512, 2048, 8192]
    #
    # for i in range(5, 15, 2):
    #     training_data_subset = training_data[:int(math.pow(2, i)), :]
    #     clf = DecisionTreeClassifier()
    #     clf.fit(training_data_subset[:, :2], training_data_subset[:, -1])
    #
    #     err = 0
    #     test_size = test_data.size / 3
    #
    #     prediction = clf.predict(test_data[:, :2])
    #     for idx, data_point in enumerate(test_data):
    #         if int(data_point[2]) != int(prediction[idx]):
    #             err += 1
    #
    #     err = err / test_size
    #     nodes = clf.tree_.node_count
    #     error.append(err)
    #     num_nodes.append(nodes)
    #
    # for idx, err in enumerate(error):
    #     print(n[idx], num_nodes[idx], err)
    #
    # plt.plot(n, error)
    # plt.xlabel("n")
    # plt.ylabel("Error err_n")
    # plt.title("n vs err_n")
    # plt.show()

    ##################################################################################################

    # For Q4 Lagrange Interpolation
    # Generate train set
    # a = 0
    # b = np.pi
    # n = 100
    #
    # train_x = np.random.uniform(a, b, n)
    # train_y = np.sin(train_x)
    # poly = lagrange(train_x, train_y)
    # np.savetxt("data/q4train.txt", np.c_[train_x, train_y])
    #
    # test_x = np.random.uniform(a, b, n)
    # test_y = np.sin(test_x)
    #
    # mse = np.sum((test_y - poly(test_x)) ** 2) / n
    # print("No noise: {}".format(math.log(mse)))
    #
    # mse = []
    #
    # for sigma in range(0, 10):
    #     mse_sum = 0
    #     for i in range(0, 99):
    #         np.random.seed()
    #         epsilon = np.random.normal(0, sigma, size=n)
    #         train_x_e = train_x + epsilon
    #         train_y_e = np.sin(test_y)
    #         poly_e = lagrange(train_x_e, train_y_e)
    #         mse_e = np.sum((test_y - poly_e(test_x)) ** 2) / n
    #         mse_sum += mse_e
    #     print("With noise: {}, stdev = {}".format(math.log(mse_sum / 10), sigma))
    #     mse.append(math.log(mse_sum / 10))
    #
    # plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], mse)
    # plt.xlabel("Standard Deviation (sigma)")
    # plt.ylabel("Log MSE Error")
    # plt.title("Standard Deviation in Zero-Mean Gaussian Noise vs Log MSE Error")
    # plt.show()
    #
    # mse = []
    #
    # for sigma in range(0, 10):
    #     mse_sum = 0
    #     for i in range(0, 99):
    #         np.random.seed()
    #         epsilon = np.random.normal(0, sigma, size=n)
    #         train_x_e = train_x + epsilon
    #         train_y_e = np.sin(test_y)
    #         poly_e = lagrange(train_x_e, train_y_e)
    #         mse_e = np.sum((test_y - poly_e(test_x)) ** 2) / n
    #         mse_sum += mse_e
    #     print("With noise: {}, stdev = {}".format(math.log(mse_sum / 10), sigma**2))
    #     mse.append(math.log(mse_sum / 10))
    #
    # plt.plot(np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) ** 2, mse)
    # plt.xlabel("Variance (sigma^2)")
    # plt.ylabel("Log MSE Error")
    # plt.title("Variance in Zero-Mean Gaussian Noise vs Log MSE Error")
    # plt.show()
