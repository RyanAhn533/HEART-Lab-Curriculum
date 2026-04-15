param_bounds = {'x1': (-1, 5),
                'x2': (0, 4)}
#x1, x2 -> 매개변수 -> 최적화된 파라미터를 찾는다 !
def y_function(x1, x2): 
    return -x1 ** 2 - (x2 - 2) ** 2 + 10

# pip install bayesian-optimization
from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f=y_function,
    pbounds=param_bounds,
    random_state=333,
)

optimizer.maximize(init_points=5,  # 'inint_points'가 아닌 'init_points'로 수정
                   n_iter=20)
#maximize = model.fit이랑 똑같음


print(optimizer.max)

