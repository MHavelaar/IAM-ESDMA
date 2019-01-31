'''
Created on 24 jan. 2019

@author: wlauping
'''
from __future__ import (division, unicode_literals, print_function, 
                        absolute_import)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ema_workbench import (TimeSeriesOutcome, perform_experiments, Constant, 
                           RealParameter, ema_logging, ScalarOutcome, load_results)

from ema_workbench.connectors.vensim import (LookupUncertainty, VensimModel, VensimModelStructureInterface)
from ema_workbench.em_framework import CategoricalParameter
from ema_workbench.em_framework.evaluators import MultiprocessingEvaluator
from ema_workbench.em_framework.parameters import Policy
from ema_workbench.util import save_results, ema_logging, CaseError


if __name__ == '__main__':
    # Turn on Logging
    ema_logging.log_to_stderr(ema_logging.INFO)
    wd = r'D:\workspace\EMAProjects\WILLEM\Afstudeerders\IAM-ESDMA-master\Model'
    model = VensimModel("EMAModel", wd=wd, model_file=wd+'\FinalModelEMA.vpm')
    model.uncertainties=[
    #Population Scenario
                         CategoricalParameter('Switch Population', (0,1,2)),
    #Land Use Scenario
                         CategoricalParameter('Switch Urbanization', (0,1,2)),
    #Economy Scenario
                         CategoricalParameter('Switch Economy', (0,1,2)),
    #Innovation Scenario
                         CategoricalParameter('Switch Innovation', (0,1,2)),
    #Climate Scenario
                         CategoricalParameter('Switch Climate Scenario', (0,1,2,3)),
    ##GENERAL UNCERTAINTIES
    #Building Mode City and Harbour                   
                         CategoricalParameter('Switch Building Mode', (0,1)),
                         CategoricalParameter('Forecast Time Forecasting Policy', (1, 2, 3, 4)),
    
    #Nature Preservation Area
                         CategoricalParameter('Switch Nature Scenario', (0,1)),
                        
    #MODEL SPECIFIC UNCERTAINTIES
                        
    #City Model
                         RealParameter('Normal Land use per house', 0.024, 0.036),
                         CategoricalParameter('Forecasting Delay Order', (1,3,5)),
                         CategoricalParameter('Planning Time Houses', (1, 2, 3, 4, 5)),
                         CategoricalParameter('Construction Time Houses', (1, 2, 3)),
                         CategoricalParameter('Delay Order House Construction', (1,3,5)),
                         RealParameter('Technical Lifetime Houses', 300, 500),
                         RealParameter('Normal Land Use per Business', 0.048, 0.072),
                         RealParameter('Jobs per harbour mtpa', 100, 150),
                         CategoricalParameter('Planning Time Business', (1, 2, 3, 4, 5)),
                         CategoricalParameter('Construction Time Business', (1,2, 3)),
                         CategoricalParameter('Delay Order Business Construction', (1,3,5)),
                         RealParameter('Technical Lifetime Business', 150, 250),
    #Harbour Model
                         CategoricalParameter('Planning Time Harbour', (1,2,3,4,5)),
                         CategoricalParameter('Construction Time Harbour', (1,2,3)),
                         CategoricalParameter('Delay Order Harbour Construction', (1,3,5)),
                         RealParameter('Technical Lifetime Harbour', 150, 250),
    #Area Model
                        ## No Model Uncertainties
    #Economy Model
                        ## No Model Uncertainties
    #Road Model
                         CategoricalParameter('Planning Time New Construction Road', (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)),
                         CategoricalParameter('Planning Time Renovation Road', (1, 2, 3)),
                         CategoricalParameter('Renovation Time Road', (0.25, 0.5, 0.75, 1)),
                         CategoricalParameter('Construction Time Road', (1, 2, 3, 4, 5)),
                         RealParameter('Closure Rate Road A10 Intervention', 0.33, 0.50),
                         RealParameter('PCE to MVT factor', 0.5, 2),
                         CategoricalParameter('Delay Time Congestion Divergence', (0.5, 0.75, 1)),
                         RealParameter('MTPA to Truck Factor', 0.000006, 0.00004 ),
    #Lock Model
                         CategoricalParameter('Delay time divergence factor', (0.5, 1.0, 1.5)),
                         CategoricalParameter('Delay time Scaling Benefits', (0.5, 1.0, 1.5)),
                         CategoricalParameter('Construction Time New Sea Lock', (2, 3, 4)),
                         CategoricalParameter('Planning Time ReOpening old Noordersluis', (2, 3, 4)),
                         CategoricalParameter('Construction Time Reopening old Noordersluis', (2, 3, 4)),
                         CategoricalParameter('Construction Time New Sea Lock No Innovation', (2, 3, 4)),
                         CategoricalParameter('Planning Time New Sea Lock No Innovation', (2, 3, 4)),
                         CategoricalParameter('Planning Time New Sea Lock Innovation', (3, 4, 5, 6)),
                         CategoricalParameter('Construction Time New Sea Lock Innovation', (3, 4, 5, 6)),
                         CategoricalParameter('Initial MTPA Leisure Vessels', (2, 3, 4))
                         ]
    
    ##GENERAL MODEL CONSTANTS
    model.constants = [
    #City Model
                       Constant('Initial Planned Houses', 5000),
                       Constant('Initial Houses', 423785), 
                       Constant('Normal size households', 1.97),
                       Constant('Initial population', 834713),
                       Constant('Labour population factor', 0.8),
                       Constant('Initial Number of jobs per business unit', 4.64),
                       Constant('Initial Planned Business', 1000),
                       Constant('Initial Business', 100000),
                       Constant('Percentage of Labour Population actively wanting a job', 0.7),
                       Constant('Labour population out of City factor', 0.07),
                
    #Harbour Model
                       Constant('Initial Planned Harbour Capacity', 50),
                       Constant('Initial Constructed Harbour Capacity', 200),
                       Constant('MTPA Canal Capacity', 500),
                       Constant('MTPA Train Capacity', 50),
                       Constant('Initial world offer sea MTPA', 95),
                       Constant('Initial world offer road MTPA', 38),
                       Constant('Initial world offer rail MTPA', 5),
                       Constant('Initial world offer canal MTPA', 63),
                       Constant('Percentage Harbour Distribution Capacity Through Sea', 0.472637),
                       Constant('Percentage Harbour Distribution Capacity Through Road', 0.189055),
                       Constant('Percentage Harbour Distribution Capacity Through Canal', 0.313433),
                       Constant('Percentage Harbour Distribution Capacity Through Train', 0.0248756),
    #Area Model
                       Constant('Initial Free Area', 25000),
                       Constant('Rail Planned', 0),
                       Constant('Canal Planned', 0),
                       Constant('Lock Planning', 0),
                       Constant('Initial Nature Area', 7500),
                       Constant('Initial Lock Area', 6),
                       Constant('Initial Canal Area', 1500),
                       Constant('Initial Rail Area', 100),
                       Constant('Initial Flooding Area', 1500),
                       Constant('Rail Demolished', 0),
                       Constant('Lock Demolition', 0),
                       Constant('Canal Demolished', 0),
                       Constant('Road Lanes Demolished', 0),
                       Constant('Flooding Area Demolished', 0),
    #Economy Model
                       Constant('Initial Local Economy', 62603475000),
                       Constant('Initial Shares Business', 0.25),
                       Constant('Initial Shares Harbour', 0.25),
                       Constant('Initial Shares Job Spending', 0.25),
    #Road Model
                       Constant('Number of Weakest Links', 1),
                       Constant('Number of Distribution Roads', 5),   
                       Constant('Initial Planned Road Capacity', 0),
                       Constant('Initial Deteriorated Road Capacity', 0),
                       Constant('Initial Diminshed Road Capacity', 0),
                       Constant('Normal Truck Percentage for calculating congestion limit', 0.15),
                       Constant('Rush Hour Car Intensity', 9000),
                       Constant('Initial IC Factor Road Intervention Moment Memory', 0),
                       Constant('Initial Intervention Moment Sewer', 0),
                       Constant('Initial Sewer System Maximum Capacity', 45),
                       Constant('Sewer Capacity increased After Intervention', 55),
                       Constant('Free Flow Capacity per Lane A10', 2000),
                       Constant('Free Flow Capacity per Lane A8', 2100),
                       Constant('Free Flow Capacity per Lane A2', 2100),
                       Constant('Free Flow Capacity per Lane A4', 2100),
                       Constant('Free Flow Capacity per Lane A5', 2100),
                       Constant('Area per Road Lane', 15),
                       Constant('Initial Number of Lanes A2', 8),
                       Constant('Initial Number of Lanes A4', 8),
                       Constant('Initial Number of Lanes A5', 4),
                       Constant('Initial Number of Lanes A10', 6),
                       Constant('Initial Number of Lanes A8', 10),
                       Constant('Delay Order Divergence', 5),
                       Constant('Reference Maximum Precipitation', 0.00414),
    #Lock Model
                       Constant('Initial Cumulative Deteriorated Lock Capacity', 0),
                       Constant('Initial Constructed Capacity New Sea Lock', 0),
                       Constant('Initial Capacity Old Middensluis', 25),
                       Constant('Initial Capacity Old Noordersluis', 66.667),
                       Constant('Initial Capacity Kleine Sluis and Zuidersluis', 8.33),
                       Constant('Initial Planned Lock Capacity', 0),
                       Constant('Demolishing Rate New Locks', 0),
                       Constant('Factor for increase required door height', 0.75833),
                       Constant('Initial Required Height for doors', 515),
                       Constant('Initial Required Height for remaining parts lock', 700),
                       Constant('Factor for increase required height remaining parts lock', 0.75833),
                       Constant('Minimum present Height of remaining parts', 725),
                       Constant('Height door Old Noordersluis', 585),
                       Constant('Height door Old Middensluis', 585),
                       Constant('IC Lock Decision Making Factor', 0.5),
                       Constant('Initial Door Height Intervention Moment Memory', 0),
                       Constant('Initial Remaining Parts Height Intervention Moment Memory', 0),
                       Constant('Initial IC Factor Lock Intervention Moment Memory', 0),
                       Constant('MTPA Increase by New Sea Lock', 175),
                       Constant('Delay Order Divergence factor', 5),
                       Constant('Delay order Scaling benefits', 5),
    #Connected Simulation
                       Constant('Switch Connected and Unconnected', 1),
    #Road Model
                       Constant('Number of Lanes Added', 2),
                       Constant('IC Decision Making Factor Road', 0.8),
                       Constant('Switch Sewer Intervention', 1),
                       Constant('Base Case Intervention Moment', 0),
                       Constant('First Intervention Moment Road System', 2017),
                       Constant('Second Intervention Moment Road System', 2019), 
    #Lock Model
                       Constant('Intervention Moment New Sea Lock', 2017),
                       Constant('Intervention Moment for 2nd New Sea Lock No Innovation', 0),
                       Constant('Intervention Moment for 2nd New Sea Lock Innovation', 0),
                       Constant('Intervention Moment for 3rd New Sea Lock No Innovation', 0),
                       Constant('Intervention Moment for 3rd New Sea Lock Innovation', 2026),
                       ]
        
    model.levers = [
                    CategoricalParameter('Switch Intervention Policy Road', (0, 1, #2, 3, 4, 5
                                                                             )),
#                     CategoricalParameter('Policy Combination Switch', (0, 1, 2, 3)),
#                     CategoricalParameter('Switch Sewer Intervention', (0, 1)),
    # Lock Model
#                     CategoricalParameter('Switch Lock Policy', (0, 1, 2, 3)),  # (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
                    ]  
    
    model.outcomes = [
#                       TimeSeriesOutcome('City Area to Total Area'),
#                       TimeSeriesOutcome('City Population'),
#                       TimeSeriesOutcome('Constructed Houses'),
#                       TimeSeriesOutcome('Constructed Business'),
#                       TimeSeriesOutcome('MTPA Distributed in Harbour'),
#                       TimeSeriesOutcome('Harbour Area to Total Area'),
#                       TimeSeriesOutcome('Available Area'),
#                       TimeSeriesOutcome('Local Economy'),
#                       TimeSeriesOutcome('Local Economic Growth'),
#                       TimeSeriesOutcome('Sell By Date for the Sewer System'),
#                       TimeSeriesOutcome('Sell By Date of the Road System Policy'),
                      TimeSeriesOutcome('Road System IC Factor'),
                      TimeSeriesOutcome('Lock System IC Factor'),
#                       TimeSeriesOutcome('Sell By Date of Door Height'),
#                       TimeSeriesOutcome('Sell By Date of Lock Remaining Parts Height'),
#                       TimeSeriesOutcome('Sell By Date of Lock System Policy IC Factor'),
#                       TimeSeriesOutcome('Jobs')
                      ]       

    nr_experiments = 1000
    n_processes = 4
    nr_policies = 2
    with MultiprocessingEvaluator(model, n_processes=n_processes) as evaluator:
        results = evaluator.perform_experiments(scenarios=nr_experiments, reporting_interval=100, policies=nr_policies, levers_sampling='ff') # 
    save_results(results, wd + '\\IAM_road_{}policies_{}runs.tar.gz'.format(nr_policies, nr_experiments))
