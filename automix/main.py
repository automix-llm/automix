from typing import Union, List, Tuple, Dict, Any, Optional, Callable, Iterable, Iterator, Set
import pandas as pd
import numpy as np
from .automix_methods import *

class Automix:
    """
    Automix class to encapsulate methods for training and evaluating 
    AutoMix Method between a Small Language Model (SLM) and Large Language Model (LLM).
    """

    def _fill_variables(self, **kwargs):
        """
        Helper to fill in default values for kwargs if they are None.
        
        Raises error if both arg and default are None.
        """

        return_args = []
        for arg_name, arg_val in kwargs.items():
            if arg_val is None:
                try:
                    return_args.append(getattr(self, arg_name)) 
                except AttributeError:
                    raise ValueError(f"Argument {arg_name} is None in both the function call and the class initialization. Please fill it in one of them")
            else:
                return_args.append(arg_val)
        return return_args


    def __init__(self,  
                 method: Union[str, Threshold],
                 slm_column: str = 'llama13b_f1',
                 llm_column: str = 'llama70b_f1',  
                 verifier_column: str = 'p_ver_13b',
                 costs: List[int] = [1, 50],
                 verifier_cost: int = 1,
                 verbose: bool = False):
        """
        Initialize Automix with parameters.
        
        Args:
            method: Automix variant, compulsory argument.
            slm_column: SLM score column name.
            llm_column: LLM score column name.
            verifier_column: Verifier confidence column name. 
            costs: Costs for [SLM, LLM].
            verifier_cost: Cost for verifier.
            verbose: Print debugging info.
        """
        self.slm_column = slm_column
        self.llm_column = llm_column
        self.verifier_column = verifier_column
        self.costs = costs
        self.verifier_cost = verifier_cost
        self.verbose = verbose

        self.method = method

        self.best_param = None

    def compute_performance_cost(self, data, to_retry, 
                                 costs=None, verifier_cost=None):
        """
        Compute overall cost and performance given retry decisions.
        
        Args:
            data: Dataset
            to_retry: Routing Decisions
            costs: Costs for SLM and LLM
            verifier_cost: Cost for verifier
        """

        costs, verifier_cost = self._fill_variables(costs=costs, verifier_cost=verifier_cost)
        slm_cost, llm_cost = costs
        
        total_cost = (~to_retry).sum() * (slm_cost) + to_retry.sum() * (llm_cost + slm_cost)
        performances = np.where(to_retry, data[self.llm_column], data[self.slm_column])
        avg_performance = performances.mean()
        avg_cost = total_cost / len(data)
        avg_cost += verifier_cost
        return avg_performance, avg_cost

    def get_slm_llm_slope_perf(self, data, 
                               costs=None, verifier_cost=None):
        """
        Compute slope of LLM vs SLM performance.
        
        Args:
            data: Dataset
            costs: Costs for SLM and LLM
            verifier_cost: Cost for verifier
        """

        costs, verifier_cost = self._fill_variables(costs=costs, verifier_cost=verifier_cost)
        slm_cost, llm_cost = costs
        
        slm_perf, llm_perf = data[self.slm_column].mean(), data[self.llm_column].mean()
        slm_llm_slope = (llm_perf - slm_perf) / (llm_cost - slm_cost)
        return slm_llm_slope, slm_perf, llm_perf

    def train(self, data, 
              costs=None, verifier_cost=None, cost_constraint = None):
        """
        Train automix by trying different parameters and picking best.
        
        Args:
            data: Dataset
            costs: Costs for SLM and LLM
            verifier_cost: Cost for verifier
        """

        costs, verifier_cost = self._fill_variables(costs=costs, verifier_cost=verifier_cost)
        slm_cost, llm_cost = costs
        
        thresh_dic = dict()
        # Some methods may requiring passing the data
        for param in self.method.generate_points(data, verifier_column=self.verifier_column):
            to_retry = self.method.run(data, param, verifier_column=self.verifier_column)
            avg_performance, avg_cost = self.compute_performance_cost(data, to_retry, costs=costs, verifier_cost=verifier_cost)

            if cost_constraint is not None:
                if avg_cost < cost_constraint[0] or avg_cost > cost_constraint[1]:
                    continue

            slm_llm_slope, slm_perf, _ = self.get_slm_llm_slope_perf(data, costs=costs, verifier_cost=verifier_cost)

            # Slope b/w automix and slm
            automix_slm_slope = (avg_performance - slm_perf) / (avg_cost - slm_cost)
            ibc_lift = (automix_slm_slope - slm_llm_slope) / slm_llm_slope
            thresh_dic[str(param)] = ibc_lift

        self.best_param = eval(max(thresh_dic, key=thresh_dic.get))
        if self.verbose:
            print('Best Param:', self.best_param, thresh_dic[str(self.best_param)])


    def infer(self, df_row):
        """
        Run trained automix on given dataframe row to get routing decision.
        
        Args:
            df_row: Dataframe row
        """

        if self.best_param is None:
            raise ValueError("Please train the model first")
            
        to_retry = self.method.run(df_row, self.best_param, verifier_column=self.verifier_column)
        return to_retry


    def evaluate(self, data, 
                 costs=None, verifier_cost=None, 
                 return_dict=False, return_decisions=False):
        """
        Evaluate trained automix on full dataframe to compute metrics.
        
        Args:
            data: Dataset
            costs: Costs for SLM and LLM
            verifier_cost: Cost for verifier
            return_dict: Return results as dict
            return_decisions: Return routing decisions
        """

        costs, verifier_cost = self._fill_variables(costs=costs, verifier_cost=verifier_cost)
        slm_cost, llm_cost = costs
        
        to_retry = data.apply(self.infer, axis=1)
        avg_performance, avg_cost = self.compute_performance_cost(data, to_retry)
        slm_llm_slope, slm_perf, llm_perf = self.get_slm_llm_slope_perf(data)
        automix_slm_slope = (avg_performance - slm_perf) / (avg_cost - slm_cost)
        ibc_lift = (automix_slm_slope - slm_llm_slope) / slm_llm_slope
        
        if return_dict:
            if return_decisions:
                return {'ibc_lift' : ibc_lift, 
                        'automix_slm_slope' : automix_slm_slope,
                        'avg_performance' : avg_performance,
                        'avg_cost' : avg_cost, 
                        'route_to_llm' : to_retry}
            else:
                return {'ibc_lift' : ibc_lift,
                        'automix_slm_slope' : automix_slm_slope,  
                        'avg_performance' : avg_performance,
                        'avg_cost' : avg_cost}

        if return_decisions:
            return ibc_lift, automix_slm_slope, avg_performance, avg_cost, to_retry
            
        return ibc_lift, automix_slm_slope, avg_performance, avg_cost