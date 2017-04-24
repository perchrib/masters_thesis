import matplotlib.pyplot as plt

# Markers and
marker = [u'+', u'o', u'o']
col = ['r', 'b', 'g']


def plot_2_classes(class1_points, class2_points):

    for x, y in class1_points:
        plt.scatter(x, y, marker=marker[0], c=col[0])

    for x, y in class2_points:
        plt.scatter(x, y, marker=marker[1], c=col[1])

    plt.show()


def plot_svm_mockup():

    class1 = [(1,1), (2,2), (2,3), (2,4), (3.5,3.5), (3,1), (3.5,1)]
    class2 = [(5.5, 4.5), (6, 5), (5, 7), (7, 4), (9, 8), (8,2)]

    plot_2_classes(class1, class2)


def plot_xor():
    class1 = [(1,0), (0,1)]
    class2 = [(0,0), (1,1)]

    plot_2_classes(class1, class2)

if __name__ == '__main__':
    # plot_svm_mockup()
    plot_xor()