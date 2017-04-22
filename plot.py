import matplotlib.pyplot as plt


def plot_svm_mockup():
    s = [u'+', u'o', u'o']
    col = ['r', 'b', 'g']

    class_1_points = [(1,1), (2,2), (2,3), (2,4), (3.5,3.5), (3,1), (3.5,1)]
    class_2_points = [(5.5, 4.5), (6, 5), (5, 7), (7, 4), (9, 8), (8,2)]

    for x, y in class_1_points:
        plt.scatter(x, y, marker=s[0], c=col[0])

    for x, y in class_2_points:
        plt.scatter(x, y, marker=s[1], c=col[1])

    # plt.plot([3.0, 6.2], [8, 0])
    plt.show()

if __name__ == '__main__':
    plot_svm_mockup()