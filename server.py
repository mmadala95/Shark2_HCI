'''

You can modify the parameters, return values and data structures used in every function if it conflicts with your
coding style or you want to accelerate your code.

You can also import packages you want.

But please do not change the basic structure of this file including the function names. It is not recommended to merge
functions, otherwise it will be hard for TAs to grade your code. However, you can add helper function if necessary.

'''

from flask import Flask, request
from flask import render_template
import matplotlib.pyplot as plt
import numpy as np
import time
import math

import json


app = Flask(__name__)

# Centroids of 26 keys
centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240, 170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50, 120]

# Pre-process the dictionary and get templates of 10000 words
words, probabilities = [], {}
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])

def line_distance(x0,y0,x1,y1):
    return math.sqrt((x1 - x0)**2 + (y1 - y0)**2)

def generate_sample_points(points_X, points_Y):



    '''Generate 100 sampled points for a gesture.

    In this function, we should convert every gesture or template to a set of 100 points, such that we can compare
    the input gesture and a template computationally.

    :param points_X: A list of X-axis values of a gesture.
    :param points_Y: A list of Y-axis values of a gesture.

    :return:
        sample_points_X: A list of X-axis values of a gesture after sampling, containing 100 elements.
        sample_points_Y: A list of Y-axis values of a gesture after sampling, containing 100 elements.
    '''

    # TODO: Start sampling (12 points)

    sample_points_X, sample_points_Y = [], []
    sample_points_X.append(points_X[0])
    sample_points_Y.append(points_Y[0])
    slope = []
    sign = []
    line_length = []
    flag = False
    flag_value = 0

    for i in range(0, len(points_X) - 1):
        line_length.append(line_distance(points_X[i], points_Y[i], points_X[i + 1], points_Y[i + 1]))
    sampling_length = sum(line_length) / 99
    # print(sampling_length)
    for i in range(0, len(points_X) - 1):
        x0 = points_X[i];
        y0 = points_Y[i];
        x1 = points_X[i + 1];
        y1 = points_Y[i + 1];

        if x0 == x1:
            curr_slope = float("inf")
            x_pos = x0
            y_pos = y0 + sampling_length

        elif x0 == x1 and y0 == y1:
            curr_slope = 0
            x_pos = x0
            y_pos = y0
        else:
            curr_slope = (y1 - y0) / (x1 - x0)

            x_pos = x0 + (0.001 * (math.sqrt(1 / (1 + (curr_slope ** 2)))))
            y_pos = y0 + (curr_slope * 0.001 * (math.sqrt(1 / (1 + (curr_slope ** 2)))))
            # x_neg = x0 - (sampling_length * (math.sqrt(1 / (1 + (curr_slope ** 2)))))
            # y_neg = y0 - (curr_slope * sampling_length * (math.sqrt(1 / (1 + (curr_slope ** 2)))))

        if (line_distance(x1, y1, x_pos, y_pos) < line_distance(x0, y0, x1, y1)):
            sign.append("+")
        else:
            sign.append("-")

        slope.append(curr_slope)
    # print (sign)
    for i in range(len(points_X) - 1):
        x0 = points_X[i];
        y0 = points_Y[i];
        x_pos = x0
        y_pos = y0
        curr_slope = slope[i]
        while (True):

            if flag:
                if (sign[i] == "+"):
                    if curr_slope == float("inf"):
                        x_pos = x_pos
                        y_pos = y_pos + flag_value
                    else:
                        x_pos = x_pos + (flag_value * (math.sqrt(1 / (1 + (curr_slope ** 2)))))
                        y_pos = y_pos + (curr_slope * flag_value * (math.sqrt(1 / (1 + (curr_slope ** 2)))))
                else:
                    if curr_slope == float("inf"):
                        x_pos = x_pos
                        y_pos = y_pos - flag_value
                    else:
                        x_pos = x_pos - (flag_value * (math.sqrt(1 / (1 + (curr_slope ** 2)))))
                        y_pos = y_pos - (curr_slope * flag_value * (math.sqrt(1 / (1 + (curr_slope ** 2)))))
                newline_length = line_distance(x_pos, y_pos, x0, y0)
                if (newline_length >= line_length[i]):
                    flag = True
                    flag_value = newline_length - line_length[i]
                    break;
                if (sign[i] != "nan"):
                    sample_points_X.append(x_pos)
                    sample_points_Y.append(y_pos)
                flag = False
                flag_value = 0
            else:
                if (sign[i] == "+"):
                    if curr_slope == float("inf"):
                        x_pos = x_pos
                        y_pos = y_pos + sampling_length
                    else:
                        x_pos = x_pos + (sampling_length * (math.sqrt(1 / (1 + (curr_slope ** 2)))))
                        y_pos = y_pos + (curr_slope * sampling_length * (math.sqrt(1 / (1 + (curr_slope ** 2)))))
                else:
                    if curr_slope == float("inf"):
                        x_pos = x_pos
                        y_pos = y_pos - sampling_length
                    else:
                        x_pos = x_pos - (sampling_length * (math.sqrt(1 / (1 + (curr_slope ** 2)))))
                        y_pos = y_pos - (curr_slope * sampling_length * (math.sqrt(1 / (1 + (curr_slope ** 2)))))

                newline_length = line_distance(x_pos, y_pos, x0, y0)
                if (newline_length >= line_length[i]):
                    flag = True
                    flag_value = newline_length - line_length[i]
                    break;
                if (sign[i] != "nan"):
                    sample_points_X.append(x_pos)
                    sample_points_Y.append(y_pos)

    if (len(sample_points_X) == 99):
        sample_points_X.append(points_X[len(points_X) - 1])
        sample_points_Y.append(points_Y[len(points_Y) - 1])

    if (len(sample_points_X) > 100):
        n = len(sample_points_X) - 100
        del sample_points_X[-n:]
        del sample_points_Y[-n:]
    # plt.scatter(sample_points_X, sample_points_Y)
    # plt.show()
    if(len(sample_points_X)==1):
        for i in range(99):
            sample_points_X.append(sample_points_X[0])
            sample_points_Y.append(sample_points_Y[0])
    return sample_points_X, sample_points_Y


# Pre-sample every template
template_sample_points_X, template_sample_points_Y = [], []
count=0
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    count+=len(X)
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)
print (count)


def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y):
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider it reasonable)
    to narrow down the number of valid words so that the ambiguity can be avoided to some extent.

    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.

    :return:
        valid_words: A list of valid words after pruning.
        valid_probabilities: The corresponding probabilities of valid_words.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''
    count=0
    threshold = 20
    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = [], [], []
    for i in range (0,len(template_sample_points_X)):

        start2start=line_distance(gesture_points_X[0][0],gesture_points_Y[0][0],template_sample_points_X[i][0],template_sample_points_Y[i][0])
        end2end=line_distance(gesture_points_X[0][len(gesture_points_X[0])-1],gesture_points_Y[0][len(gesture_points_Y[0])-1],template_sample_points_X[i][len(template_sample_points_X[i])-1],template_sample_points_Y[i][len(template_sample_points_Y[i])-1])
        sum=start2start+end2end
        # print(sum)
        if(sum<threshold):
            valid_words.append(words[i])
            valid_template_sample_points_X.append(template_sample_points_X[i])
            valid_template_sample_points_Y.append(template_sample_points_Y[i])
            count+=1

    # print ("c",count)
    print (valid_words)

    # TODO: Set your own pruning threshold

    # TODO: Do pruning (12 points)

    return valid_words, valid_template_sample_points_X, valid_template_sample_points_Y

def get_normalising_factor(L,x,y):
    width=max(x)-min(x)
    height=max(y)-min(y)
    if(width-height==0):
        return L
    return (L/max(width,height))

def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the shape score for every valid word after pruning.

    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param valid_template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of shape scores.
    '''
    # print(valid_template_sample_points_X[6])

    shape_scores = []
    # TODO: Set your own L
    L = 1

    # TODO: Calculate shape scores (12 points)
    # print (len(gesture_sample_points_X[0]))
    # print(len(valid_template_sample_points_X))
    # print (len(valid_template_sample_points_X[0]))
    # print(len(valid_template_sample_points_X[1]))
    avg_gesture_x=sum(gesture_sample_points_X)/len(gesture_sample_points_X)
    avg_gesture_y=sum(gesture_sample_points_Y)/len(gesture_sample_points_Y)
    gesture_sample_points_X1=[val - avg_gesture_x for val in gesture_sample_points_X]
    gesture_sample_points_Y1=[val - avg_gesture_y for val in gesture_sample_points_Y]
    s_gesture=get_normalising_factor(L,gesture_sample_points_X1,gesture_sample_points_Y1)
    # print(s_gesture)
    gesture_sample_points_X1= [val * s_gesture for val in gesture_sample_points_X1]
    gesture_sample_points_Y1= [val * s_gesture for val in gesture_sample_points_Y1]


    for i in range (0,len(valid_template_sample_points_X)):
        avg_template_x = sum(valid_template_sample_points_X[i]) / len(valid_template_sample_points_X[i])
        avg_template_y = sum(valid_template_sample_points_Y[i]) / len(valid_template_sample_points_Y[i])
        valid_template_sample_points_X1 = [val - avg_template_x for val in valid_template_sample_points_X[i]]
        valid_template_sample_points_Y1 = [val - avg_template_y for val in valid_template_sample_points_Y[i]]
        s_template = get_normalising_factor(L, valid_template_sample_points_X1, valid_template_sample_points_Y1)
        valid_template_sample_points_X1 = [val * s_template for val in valid_template_sample_points_X1]
        valid_template_sample_points_Y1 = [val * s_template for val in valid_template_sample_points_Y1]


    for i in range (0,len(valid_template_sample_points_X)):
        eucledian_distance=0
        for j in range(0,len(valid_template_sample_points_X[i])):
            eucledian_distance+=line_distance(gesture_sample_points_X1[j],gesture_sample_points_Y1[j],valid_template_sample_points_X1[j],valid_template_sample_points_Y1[j])
        # print(eucledian_distance)
        shape_scores.append(eucledian_distance/len(gesture_sample_points_X))

    # print(shape_scores)
    return shape_scores

def get_d(p_i_x,p_i_y,q_x,q_y):
    difference=[]
    for i in range(len(q_x)):
        difference.append(line_distance(p_i_x,p_i_y,q_x[i],q_y[i]))
    return min(difference)

def get_D(u_x,u_y,t_x,t_y,r):
    value=0
    for i in range(len(u_x)):
        value+=max(get_d(u_x[i],u_y[i],t_x,t_y)-r, 0)
    return value


def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the location score for every valid word after pruning.

    In this function, we should compare the sampled user gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a location score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of location scores.
    '''
    location_scores = []
    radius = 15
    alpha= [0.01 for i in range (0,100)]


    for i in range (len(valid_template_sample_points_X)):
        value1=get_D(gesture_sample_points_X,gesture_sample_points_Y,valid_template_sample_points_X[i],valid_template_sample_points_Y[i],radius)
        value2=get_D(valid_template_sample_points_X[i],valid_template_sample_points_Y[i],gesture_sample_points_X,gesture_sample_points_Y,radius)

        if (value1 == 0 and value2 == 0):
            location_scores.append(0)
        else:
            score=0
            for j in range (len(valid_template_sample_points_X[i])):
                distance=line_distance(gesture_sample_points_X[j],gesture_sample_points_Y[j],valid_template_sample_points_X[i][j],valid_template_sample_points_Y[i][j])
                score+=alpha[j]*distance
        # print(score)
            location_scores.append(score)


    # TODO: Calculate location scores (12 points)
    # print(location_scores)
    return location_scores


def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    # TODO: Set your own shape weight
    shape_coef = 0.5
    # TODO: Set your own location weight
    location_coef = 0.5
    # print(shape_scores)
    # print(location_scores)
    for i in range(len(shape_scores)):
        integration_scores.append(shape_coef * shape_scores[i] + location_coef * location_scores[i])
    # print(integration_scores)
    return integration_scores


def get_best_word(valid_words, integration_scores):
    '''Get the best word.

    In this function, you should select top-n words with the highest integration scores and then use their corresponding
    probability (stored in variable "probabilities") as weight. The word with the highest weighted integration score is
    exactly the word we want.

    :param valid_words: A list of valid words.
    :param integration_scores: A list of corresponding integration scores of valid_words.
    :return: The most probable word suggested to the user.
    '''
    # best_word = 'the'
    # print(integration_scores)

    if (len(integration_scores)!=0):
        scores=np.array(integration_scores)
        best_scores=scores.argsort()[:3]
        best_word=""
        for i in range (len(best_scores)):
            best_word=best_word+" "+valid_words[best_scores[i]]



        # best_word=valid_words[integration_scores.index(min(integration_scores))]
    else:
        best_word="No word found"

    # TODO: Set your own range.
    n = 3
    # TODO: Get the best word (12 points)
    print("Best 3: ",best_word)
    return best_word


@app.route("/")
def init():
    return render_template('index.html')


@app.route('/shark2', methods=['POST'])
def shark2():

    start_time = time.time()
    data = json.loads(request.get_data())

    gesture_points_X = []
    gesture_points_Y = []
    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])
    gesture_points_X = [gesture_points_X]
    gesture_points_Y = [gesture_points_Y]


    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X[0], gesture_points_Y[0])

    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y)

    shape_scores = get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    integration_scores = get_integration_scores(shape_scores, location_scores)

    best_word = get_best_word(valid_words, integration_scores)

    end_time = time.time()

    return '{"best_word":"' + best_word + '", "elapsed_time":"' + str(round((end_time - start_time) * 1000, 5)) + 'ms"}'


if __name__ == "__main__":
    app.run()
