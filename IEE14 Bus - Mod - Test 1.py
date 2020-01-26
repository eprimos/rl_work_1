#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import warnings

# Removing deprecated mthods message
warnings.filterwarnings("ignore", category=UserWarning)

# The load to be restored
goal = 6

# Map the edges to the elements of the grid
points_list = [(0,1),
               (1,2),(2,3),(2,4),(2,5),(2,9),(2,10),
               (1,5),(5,4),(5,6),
               (3,8),(3,11),
               (4,7),(4,3)]
               
# how many points in graph?
MATRIX_SIZE = len(points_list)

# Set the names of the edges
G=nx.Graph()
G.add_edges_from(points_list)
mapping={0:'0 - GU1',
         1:'1 - BUS1',
         2:'2 - BUS2',
         3:'3 - BUS3',
         4:'4 - BUS4',
         5:'5 - BUS5',
         6:'6 - LDL5',
         7:'7 - LDL4',
         8:'8 - LDL3',
         9:'9 - LDL2',
         10:'10 - GU2',
         11:'11 - GU3'
        }

H=nx.relabel_nodes(G,mapping) 

pos = nx.spring_layout(H)

nx.draw_networkx_nodes(H,pos)
nx.draw_networkx_edges(H,pos)
nx.draw_networkx_labels(H,pos)

#print(len(points_list))
#print(MATRIX_SIZE)
plt.show();


# create matrix x*y
R = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))
R *= -1

# assign zeros to paths and 100 to goal-reaching point
for point in points_list:
    if point[1] == goal:
        R[point] = 100
    else:
        R[point] = 0

    if point[0] == goal:
        R[point[::-1]] = 100
    else:
        # reverse of point
        R[point[::-1]]= 0

# add goal point round trip
R[goal,goal]= 100

#print(R)

Q = np.matrix(np.zeros([MATRIX_SIZE,MATRIX_SIZE]))

# learning parameter
gamma = 0.8

initial_state = 1

def available_actions(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act

def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_act,1))
    return next_action

def update(current_state, action, gamma):
    
  max_index = np.where(Q[action,] == np.max(Q[action,]))[1]
  
  if max_index.shape[0] > 1:
      max_index = int(np.random.choice(max_index, size = 1))
  else:
      max_index = int(max_index)
  max_value = Q[action, max_index]
  
  Q[current_state, action] = R[current_state, action] + gamma * max_value
#  print('max_value', R[current_state, action] + gamma * max_value)
  
  if (np.max(Q) > 0):
    return(np.sum(Q/np.max(Q)*100))
  else:
    return (0)

available_act = available_actions(initial_state) 
action = sample_next_action(available_act)

update(initial_state, action, gamma)


# Training
scores = []
for i in range(1500):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(current_state)
    available_act_len = len(available_act)
    
    if ( available_act_len > 0):
       action = sample_next_action(available_act)
    score = update(current_state,action,gamma)
    scores.append(score)
#    print ('Score:', str(score))
    
#print("Trained Q matrix:")
#print(Q/np.max(Q)*100)


# Testing
# Point to the initial edge (Generating Unit)
current_state = 11
steps = [current_state]

while current_state != goal:

    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
    
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size = 1))
    else:
        next_step_index = int(next_step_index)
    
    steps.append(next_step_index)
    current_state = next_step_index

print("Most efficient path:")
print(steps)

plt.plot(scores)
plt.show();