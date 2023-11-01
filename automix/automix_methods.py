from typing import Union, List, Tuple, Dict, Any, Optional, Callable, Iterable, Iterator, Set
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
	

class Threshold:

	def __init__(self, num_bins, **kwargs):
		# Set all kwargs to self
		for key, value in kwargs.items():
			setattr(self, key, value)   

		self.num_bins = num_bins
		self.gap = 1/num_bins
		pass

	def run(self, data : pd.DataFrame, threshold : float = 0.5, verifier_column = 'p_ver_13b') -> pd.DataFrame:
		to_retry = data[verifier_column] <= threshold

		return to_retry
	
	def generate_points(self, data = None, verifier_column = 'p_ver_13b'):
		gap = 1/self.num_bins
		return [x*gap for x in range(self.num_bins+1)]

class DoubleThreshold(Threshold):

	def run(self, data : pd.DataFrame, threshold : Tuple[float, float] = [0.25, 0.75], verifier_column = 'p_ver_13b') -> pd.DataFrame:
		try:
			to_retry = data[verifier_column].between(threshold[0], threshold[1])
		except:
			to_retry = data[[verifier_column]].between(threshold[0], threshold[1])[verifier_column]

		return to_retry

	def generate_points(self, data = None, verifier_column = 'p_ver_13b'):
		points = []
		for i in range(self.num_bins):
			for j in range(i+1, self.num_bins):
				points.append((i*self.gap, j*self.gap))
		return points

class TripleThreshold(DoubleThreshold):	

	def run(self, data : pd.DataFrame, threshold : Tuple[float, float, float] = [0.25, 0.5, 0.75], verifier_column = 'p_ver_13b') -> pd.DataFrame:
		try:
			to_retry = data[verifier_column].between(0, threshold[0]) | data[verifier_column].between(threshold[1], threshold[2])
		except:
			to_retry = data[[verifier_column]].between(0, threshold[0])[verifier_column] | data[[verifier_column]].between(threshold[1], threshold[2])[verifier_column]
		return to_retry

	def generate_points(self, data = None, verifier_column = 'p_ver_13b'):
		points = []
		for i in range(self.num_bins):
			for j in range(i+1, self.num_bins):
				for k in range(j+1, self.num_bins):
					points.append((i*self.gap, j*self.gap, k*self.gap))
		return points

class SelfConsistency(Threshold):

	def generate_points(self, data = None, verifier_column = 'p_ver_13b'):
		return [0.5]

class POMDPSimple:

	def compute_obs_probs(self, df, verifier_column = 'p_ver_13b'):
		categories = ['NEEDY', 'GOOD', 'HOPELESS']
		obs_probs = np.zeros((self.num_bins+1, 3))

		for idx, prob in enumerate([i*self.gap for i in range(self.num_bins+1)]):
			df_new = df[df[verifier_column] - prob < self.gap/2] 
			df_new = df_new[df_new[verifier_column] - prob > -self.gap/2] 
			try:    
				vcs = df_new['category'].value_counts()
				obs_probs[idx] = [(vcs[cat] if cat in vcs else 0)/len(df_new) for cat in categories]
			except Exception as e:
				# print('Error in categorization', e)
				...
				
		belief = np.array([1 for _ in range(len(categories))])
		if self.init_belief:
			# Use value_counts from category to get initial belief
			belief = np.array([df['category'].value_counts()[cat] if cat in df['category'].value_counts() else 0 for cat in categories])

		belief = belief/sum(belief)

		action_seqs  = []

		for reward in range(5000):
			action_array = np.array([[-reward, 0, -reward-1],[-100, -100, -reward-100]])
			actions = []
			for i in range(self.num_bins+1):
				scores = obs_probs[i]*action_array
				actions.append(np.argmax(scores.sum(axis=1)))
			action_seqs.append(tuple(actions))

		return list(set(action_seqs))

	def get_neearest_prob_idx(self, prob):
		for i in [self.gap * i for i in range(self.num_bins+1)]:
			if prob - i < self.gap/2 and prob - i >= -self.gap/2:
				return int(i//self.gap)

	def get_action(self, x, action_seq):
		return action_seq[self.get_neearest_prob_idx(x)] == 1

	def __init__(self, num_bins, init_belief = False, **kwargs) -> None:

		# Set all kwargs to self
		for key, value in kwargs.items():
			setattr(self, key, value)   

		self.num_bins = num_bins
		self.gap = 1/num_bins
		self.init_belief = init_belief

	def run(self, data : pd.DataFrame, action_seq : List[int] = [], verifier_column = 'p_ver_13b') -> pd.DataFrame:
		if isinstance(data[verifier_column], float):
			to_retry = data[[verifier_column]].apply(lambda x: self.get_action(x, action_seq))[verifier_column]
		else:
			to_retry = data[verifier_column].apply(lambda x: self.get_action(x, action_seq)) 
		return to_retry

	def generate_points(self, data, verifier_column = 'p_ver_13b'):
		return self.compute_obs_probs(data, verifier_column = verifier_column)

class GreedyPOMDP(POMDPSimple):

	def __init__(self, num_bins, init_belief = False, slm_column = 'llama13b_f1', llm_column = 'llama70b_f1', **kwargs) -> None:
		super().__init__(num_bins, init_belief, **kwargs)
		self.slm_column = slm_column
		self.llm_column = llm_column

	def generate_points(self, data = None, verifier_column = 'p_ver_13b'):
		points = []
		for x in [i for i in range(self.num_bins+1)]:
			df_fil = data[data[verifier_column].apply(lambda y: self.get_neearest_prob_idx(y) == x)]
			df_fil = data[data[verifier_column].apply(lambda y: y//self.gap == x)]
			
			wt = len(df_fil)/len(data)
			delta_f = df_fil['llama70b_f1'] - df_fil['llama13b_f1']
			points.append((delta_f.mean() if len(df_fil)!=0 else 0, x, wt))
		
		points = list(zip(gaussian_filter([x[0] for x in points], sigma=2), [x[1] for x in points], [x[2] for x in points]))				

		params = [[0 for _ in range(self.num_bins+1)] for _ in range(len(points))]
		means = []
		total_perf = 0
		total_cost = 1

		for i, index in enumerate(sorted(points[:], key = lambda x: x[0])[::-1]):
			if i == 0: params[i][index[1]] = 1
			else:
				idx = index[1]
				params[i] = params[i-1].copy()
				params[i][idx] = 1
			total_perf += index[0]*index[2]
			total_cost += index[2] * 49
		return params
		

class AutomixUnion:

	# Performs union of two methods

	def __init__(self, *methods):
		self.methods = list(methods)

	def run(self, data : pd.DataFrame, param, verifier_column = 'p_ver_13b') -> pd.DataFrame:
		return self.methods[param[1]].run(data, param[0], verifier_column=verifier_column)

	def generate_points(self, data = None, verifier_column = 'p_ver_13b'):
		new_points = []
		for i, meth in enumerate(self.methods):
			new_points.extend([(x,i) for x in meth.generate_points(data, verifier_column=verifier_column)])
		return new_points

	def __repr__(self) -> str:
		return 'AutomixUnion(' + ', '.join([str(x) for x in self.methods]) + ')'


class FixedAnswerRouting:
	
	def __init__(self, method, fixed_routing_elems : List[str] = [], ans_column = 'llama13b_pred_ans'):
		self.fixed_routing_elems = fixed_routing_elems
		self.method = method
		self.ans_column = ans_column

	def run(self, data : pd.DataFrame, param, verifier_column = 'p_ver_13b') -> pd.DataFrame:
		if isinstance(data[self.ans_column], str):
			to_retry = data[[self.ans_column]].apply(lambda x: x in self.fixed_routing_elems)[self.ans_column]
		else:
			to_retry = data[self.ans_column].apply(lambda x: x in self.fixed_routing_elems) 
		to_retry = to_retry | self.method.run(data, param, verifier_column=verifier_column)
		return to_retry

	def generate_points(self, data = None, verifier_column = 'p_ver_13b'):
		return self.method.generate_points(data, verifier_column=verifier_column)

	def __repr__(self) -> str:
		return 'FixedAnswerRouting(' + str(self.method) + ', ' + str(self.fixed_routing_elems) + ')'


# Since it is a single step POMDP, we do not require explicit use of POMDP sovlers.
POMDP = lambda *args, **kwargs: AutomixUnion(POMDPSimple(*args, **kwargs), GreedyPOMDP(*args, **kwargs), DoubleThreshold(*args, **kwargs), Threshold(*args, **kwargs))