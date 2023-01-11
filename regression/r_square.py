from statistics import mean
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

##xs = np.array([1,2,3,4,5,6], dtype=np.float64)
##ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def dataset(n,variance,step=2,correlation=False):
    val=1
    ys=[]
    for i in range(n):
        y=val+random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation =='pos':
            val+=step
        elif correlation and correlation == 'neg':
            val_=step
    xs=[i for i in range(len(ys))]

    return np.array(xs,dtype=np.float64),np.array(ys,dtype=np.float64)

def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    b = mean(ys) - m*mean(xs)
    return m, b

def squared_error(ys_original,ys_line):
    return sum((ys_line - ys_original)*(ys_line - ys_original))

def r_squared(ys_original,ys_line):
    y_mean=[mean(ys_original) for y in ys_original]
    squared_error_regresion = squared_error(ys_original,ys_line)
    squared_error_mean = squared_error(ys_original,y_mean)
    return 1 - (squared_error_regresion/squared_error_mean)

xs,ys = dataset(60,30,2,correlation='pos')

m, b = best_fit_slope_and_intercept(xs,ys)
regression_line = [(m*x)+b for x in xs]

predict_x=6
predict_y=(m*predict_x)+b

accuracy = r_squared(ys,regression_line)
print(accuracy)

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,color='g')
plt.plot(xs,regression_line)
plt.show()
