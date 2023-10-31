import math
import string
import numpy as np


# Naive Bayes
def get_count_prob(lang, range, out=False):
    counts = dict.fromkeys(string.ascii_lowercase + " ", 0)
    total = 0
    for i in range:
        with open(f'data\\documents\\{lang}{i}.txt', encoding='utf-8') as f:
            for line in f:
                for char in line:
                    c = char.lower()
                    if c in counts:
                        counts[c] += 1
                        total += 1
    prob = dict.fromkeys(string.ascii_lowercase + " ", 0)
    if out:
        print(f"{lang}:")
    for key in counts:
        if out:
            print(f"{key} & {(counts[key] + (1 / 2)) / (total + (27 / 2))}")
        prob[key] = math.log((counts[key] + (1 / 2)) / (total + (27 / 2)), math.e)
    return counts, prob


def multinomial_prob(theta, x):
    p = 0
    for key in x:
        p += (x[key] * theta[key])
    return p


def predict(x_e, x_j, x_s):
    if x_e > x_j and x_e > x_s:
        print("Prediction: e")
        return 0
    elif x_j > x_e and x_j > x_s:
        print("Prediction: j")
        return 1
    elif x_s > x_e and x_s > x_j:
        print("Prediction: s")
        return 2
    return -1


if __name__ == '__main__':
    # Q3.1
    prior = math.log(1 / 3)

    # Q3.2
    _, e_theta = get_count_prob("e", range(0, 10), out=True)

    # Q3.3
    _, j_theta = get_count_prob("j", range(0, 10), out=True)
    _, s_theta = get_count_prob("s", range(0, 10), out=True)

    # Q3.4
    x, _ = get_count_prob("e", range(10, 11))
    print("x vector:")
    for key in x:
        print(f"{key} & {x[key]}")

    # Q3.5
    x_e = multinomial_prob(e_theta, x)
    x_j = multinomial_prob(j_theta, x)
    x_s = multinomial_prob(s_theta, x)
    print(x_e)
    print(x_j)
    print(x_s)

    # Q3.6
    pred = predict(x_e, x_j, x_s)

    # Q3.7
    confusion_matrix = [[0] * 3, [0] * 3, [0] * 3]

    for i in range(10, 20):
        e_x, _ = get_count_prob("e", range(i, i + 1))
        e_x_e = multinomial_prob(e_theta, e_x)
        e_x_j = multinomial_prob(j_theta, e_x)
        e_x_s = multinomial_prob(s_theta, e_x)
        e_pred = predict(e_x_e, e_x_j, e_x_s)
        confusion_matrix[0][e_pred] += 1

        j_x, _ = get_count_prob("j", range(i, i + 1))
        j_x_e = multinomial_prob(e_theta, j_x)
        j_x_j = multinomial_prob(j_theta, j_x)
        j_x_s = multinomial_prob(s_theta, j_x)
        j_pred = predict(j_x_e, j_x_j, j_x_s)
        confusion_matrix[1][j_pred] += 1

        s_x, _ = get_count_prob("s", range(i, i + 1))
        s_x_e = multinomial_prob(e_theta, s_x)
        s_x_j = multinomial_prob(j_theta, s_x)
        s_x_s = multinomial_prob(s_theta, s_x)
        s_pred = predict(s_x_e, s_x_j, s_x_s)
        confusion_matrix[2][s_pred] += 1

    print(confusion_matrix)

