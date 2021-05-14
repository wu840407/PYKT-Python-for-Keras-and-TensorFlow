import tensorflow as tf
from numpy import array

v1 = [3.0, 4.0, 5.0]

s1 = tf.nn.softmax(v1)
print("original result=", array(v1) / array(v1).sum())
print("softmax result=", s1.numpy())

type(grid_result)

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, std, param in zip(means, stds, params):
    print(f"{param}==>mean={mean:.3f}, std={std:.3f}")

f"Best param={grid_result.best_params_}, score={grid_result.best_score_}"