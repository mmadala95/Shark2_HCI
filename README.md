Sampling:
    Initially calculated the sign of every line(between 2 points) to get whether i have take + of - for next equidistant point
    Used the formula:
            x=x0 +- (requiredDistance* sqrt(1/1+ slope^2))
            y=y0 +- (slope*requiredDistance* sqrt(1/1+ slope^2))
    validated if the equidistant point cross the current line , then carried it to next line.
    Generated 1000000 points for sample template dataset
    Extra functions: wrote a function called line_distance to calculate distance between two points.

Pruning:
    Calculated the start to start and end to end distance between gesture word and dataset words.
    Eliminated words whose length id greater than threshold(20).
    returned valid words ,valid template words points

ShapeScores:
    First calculated the mean of gesture x and y.
    Subtracted mean from each value of gesture.
    calculated normalising factor for it using s = L / max(W , H) where W = max(x)-min(x) and H = max(y)-min(y)
    Updated each of gesture values by multiplying with normalising factor.
    Performed the same for all valid template words
    Finally calculated point to point distance between gesture and each template word and divided by length : 1/N * sum(||g_i - t_i||2)
    Extra functions: wrote getNormalising factor to get s.

LocationScores:
    assigned alpha to 0.01 for each position so that sum=1
    for delta
    calculated D and d using formula given in paper
    if D(g,t) nad D(t,g) for a word is same and equal to 0 then set location score to 0
    if not calculated point to point distance for gesture and template word , then multiplied with corresponding alpha and set as location score
    Extra functions: wrote function for D and d calculation

IntegrationScores:
    set coefficient for shape and location as 0.5 and returned the sum.

getBestWord:
    used numpy
    returned three words in the order if least integration scores

GoogleDrive link : https://drive.google.com/drive/folders/1HIDDAYiz2XGRjqdSrgV8idkzglvnu4l7?usp=sharing





