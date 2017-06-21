import sklearn as sk
import numpy as np
import pydot as pd
from sklearn.ensemble import RandomForestRegressor
import sklearn.tree as sk_tree

house_sales = np.genfromtxt('../data/raw_test.dat', delimiter=',',dtype=int)

regressor = RandomForestRegressor(n_estimators=2, min_samples_split=1)
print(house_sales)
regressor.fit(house_sales[:,1:], house_sales[:,0])


#training using the built-in cross-validation approach
#train single decision tree
print('training single decision tree...\n')
dtree = sk_tree.DecisionTreeRegressor(random_state=0, min_samples_leaf=1)
dtree = dtree.fit(house_sales[:,1:], house_sales[:,0])
print('training score: ' + str(dtree.score(house_sales[:,1:], house_sales[:,0])))

#export visualisation of tree to dotfile
sk_tree.export_graphviz(dtree, out_file='treepic.dot') #produces dot file

#convert dot file to png
graph = pd.graph_from_dot_file('treepic.dot')
graph.write_png('treepic.png')

