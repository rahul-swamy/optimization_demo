import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

import time
import csv
import numpy as np
import matplotlib.pyplot as plt
font = {'family' : 'serif',
        # 'weight' : 'bold',
        'size'   : 30}
plt.rc('font', **font)
import networkx as nx
import itertools
import sys
import json
import random
from heapq import nlargest
import math
import operator
import bisect
import maxcardinalitymatching
from functools import reduce
import gerrychain
# random.seed(a=12454549)
import visualize_gis
import os
import pandas as pd
import generate_initial_solution_and_optimize_simple as draw_map
start_time_full = time.time()
from PIL import Image
import AZ_blocklevel_algo
import pickle
import SessionState


# st.set_option('deprecation.showPyplotGlobalUse', False)
#
# # session_state = SessionState.get(random_number=random.random())
# session = st.session_state.get(random.random())
# if st.button('Increment'):
#     session.counter += 1
#
# st.info(f'The random number is {session}')
# # st.info(f'The counter is {session.counter}')
# # st.write("This number should be unique for each browser tab:", session_state.random_number)

st.write("""
Copyright © 2022 Institute for Computational Redistricting (ICOR), University of Illinois
# Optimap: An Optimization Tool for Congressional Redistricting

Scroll to the bottom of this page to create Arizona's nine congressional districts to optimize for geographical *compactness* and political *competitiveness*.
""")



#### Why compactness and competitiveness?



# #### Terminology
#
# - *Majority-minority district*: a district whose majority population is composed of a racial minority.
# - *Compactness*: the shape of the districts measured by the total perimeter of the districts; the lesser the perimeter the more "compact" the map looks.
# - *Competitive district*: when a district's political lean is less than 7% as defined by AZ-IRC.
# - *Coarsening levels*: the pre-processing stage of our algorithm creates "levels" of smaller inputs to improve computational speed; larger the number of levels, the faster the algorithm is.
# - *Local search*: an algorithm to iteratively optimize a metric (such as compactness or competitiveness) in multiple steps. Each step tweaks a given map by re-drawing the boundary between two randomly selected districts.


# #### Data sources
# - Geographical, population and demographic data from Census 2020.
# - Political affiliations from nine federal and state-level elections in 2016 and 2020.
# - Preserved communities from the Arizona Independent Redistricting Commission (AZ-IRC)'s website.


# @st.cache
def user_input_features():
    # K = 9 # Number of districts
    # n_counties_to_break = 7 #Number of counties to break out of 82
    # eps = .01 # Population balance deviation for stage 1
    # number_of_majority_minority_dists_needed

    st.sidebar.header('Choose input parameters')

    # K = st.sidebar.slider('Number of districts', 5,10,9)
    # n_counties_to_break = st.sidebar.slider('Number of counties to divide', 7,15,8)
    eps = st.sidebar.slider('Population deviation threshold', 0.005, 0.05, 0.02)
    number_of_majority_minority_dists_needed = st.sidebar.slider('Number of majority-minority districts', 0,2,1)

    st.sidebar.header('Choose performance parameters')
    n_levels = st.sidebar.slider('Number of levels of coarsening', 0,3,2)
    n_iterations = st.sidebar.slider('Max. number of local search steps', 100,5000,1000)
    n_iterations_no_improvement = st.sidebar.slider('Max. number of steps with no improvement', 100,500,200)

    data = {
            # '# districts': K,
            # '# counties to divide': n_counties_to_break,
            'Population deviation threshold': eps,
            'Number of maj-min districts': number_of_majority_minority_dists_needed}
    features = pd.DataFrame(data, index=[0])
    return features, n_iterations, n_iterations_no_improvement, n_levels

df,n_iterations, n_iterations_no_improvement, n_levels = user_input_features()
# K = df['# districts'][0]
K = 9
# n_levels = 2
# n_counties_to_break = df['# counties to divide'][0]
n_counties_to_break = 7
eps = df['Population deviation threshold'][0]
number_of_majority_minority_dists_needed = df['Number of maj-min districts'][0]
draw_map.eps = eps
# st.subheader('User Input parameters')
# st.write(df)
county_graph, P_bar_county = draw_map.input.read_data_county_2020(4)
tract_graph, P_bar_tract = draw_map.input.read_data_tract_2020(4)
P_bar = sum(tract_graph.nodes[i]['pop'] for i in tract_graph.nodes())/K
# block_graph, P_bar_block = input.read_data_block_2020(4)
communities_dict, places_dict = draw_map.input.read_communities_of_interest_and_places()
hybrid_tract_graph, label_community_tract, label_tract_community = draw_map.multilevel_algo.create_hybrid_graph(county_graph, tract_graph, n_counties_to_break, communities_dict, places_dict,'tract')
fraction_dem = round(100*sum(tract_graph.nodes[i]['p_dem'] for i in tract_graph.nodes())/sum(tract_graph.nodes[i]['p_dem']+tract_graph.nodes[i]['p_rep'] for i in tract_graph.nodes()),2)
st.write('Arizona has',int(P_bar*K),"people residing in",len(tract_graph.nodes()),'census tracts,',len(county_graph.nodes()),'counties,', len(communities_dict),'preserved communities, and',len(places_dict),'preserved townships. ' \
    'The Democratic and Republican parties have',fraction_dem,'% and ',100-fraction_dem,'% support, respectively, based on nine elections in 2018 and 2020. ' \
    'The hispanic minority population constitute',round(100*sum(tract_graph.nodes[i]['p_hispanic'] for i in tract_graph.nodes())/sum(tract_graph.nodes[i]['pop'] for i in tract_graph.nodes()),2),'% of Arizona\'s population based on Census 2020.')
st.write('Beyond compactness and competitiveness, as mandated by the US and Arizona Constitutions, we want to draw districts that are contiguous, balanced in population, preserve communities of interest, and protect minority (hispanic) representation.')

# st.write("Percentage of Hispanic population from Census 2020:",round(100*sum(tract_graph.nodes[i]['p_hispanic'] for i in tract_graph.nodes())/sum(tract_graph.nodes[i]['pop'] for i in tract_graph.nodes()),2),'%.')
# st.write('Ideal population has',int(P_bar),'people.')


image = Image.open('fig_data.png')
st.image(image, caption='Demgraphic, racial and political data from Census 2020 and past elections.')


st.write("""
#### Why compactness and competitiveness

The Arizona Constitution requires that districts are *compact* and *competitive* to the extent practicable.
**Compactness** is a measure of how well-rounded the shapes of the districts are.
Optimizing compactness creates districts where voters are geographically proximal to each other.
However, purely optimizing for compactness could unintentionally create political disadvantage, for example, when urban voters are \"packed\" into a fewer districts.
Hence, we optimize for both compactness and the number of **competitive** districts to create geographically compact districts without favoring either party too much.
""")

st.info('**Compactness** is measured by the total perimeter length of all the districts.',icon="ℹ️")
st.info('A **competitive district** is one where neither party has more than 57% party support (as defined by Arizona Independent Redistricting Commission).',icon="ℹ️")

st.write("""
#### How Optimap works
The algorithm is powered by a *multilevel algorithm* and a series of *local search algorithms*.
Starting from the census tracts, the **multilevel algorithm** creates *levels* of progressively *coarsened* inputs.
More the levels, faster is the algorithm.

The **local search algorithm** iteratively optimizes a metric (such as compactness or competitiveness) in multiple steps. Each step tweaks a given map by re-drawing the boundary between two randomly selected districts.

**Stage 1** creates a initial district plan that is contiguous, balanced in population, and has the requisite number of majority-minority districts. **Stage 2** optimizes for compactness. **Stage 3** optimizes for political competitiveness, while retaining a reasonable amount of compactness.

#### How to use this tool
Select parameter values on the left panel and click the button at the bottom of this page. Carefully chosen parameter values influence the algorithm *speed* and the map *quality*.
- *Population deviation threshold*: a value between 0.01 and 0.05 for the maximum fractional deviation in district population from the ideal district population.
- *Number of majority-minority districts*: number of districts where the Hispanic minority forms the majority population.
- *Number of levels of coarsening*: a value between 0 and 2 signifying the extent of coarsening; larger the number of levels, faster  is the algorithm.
- *Max. number of local search steps*: number of local search steps for termination; longer it runs, better is the solution.
- *Max. number of steps with no improvement*: number of local search steps with no improvement in compactness or competitiveness for termination; longer we wait, better is the solution.
""")


coarse_graph, coarse_graph_before_aggregation, level_l_to_l_plus_1, level_l_plus_1_to_l, P_bar, level_graph_l = draw_map.multilevel_algo.aggregate_multilevel(tract_graph, hybrid_tract_graph, label_community_tract, n_levels, 1, plot_coarse_levels=False)
# def user_input_features2():

#     L = st.sidebar.slider('Number of levels of coarsening', 0,1,2)
#     data = {'L': L}
#     features = pd.DataFrame(data, index=[0])
#     return features



draw_maps_button = st.button('Click here to draw the maps')

if draw_maps_button:

# if st.button('Draw initial map'):

    st.subheader('Stage 1: A starting map')
    with st.spinner('Generating an initial district map...'):
        start_time_initial_heuristic = time.time()
        z_i, z_k = draw_map.algorithm.vickrey_initial_solution_heuristic(coarse_graph)
        time_vickrey = time.time() - start_time_initial_heuristic
    st.success('Done creating a contiguous and balanced map!')


    z_i_hybrid, z_k_hybrid, objbest_iterations_alllevels, obj_best_final = draw_map.algorithm.uncoarsen_with_local_search(z_i, z_k, level_l_to_l_plus_1, level_l_plus_1_to_l, county_graph, tract_graph, level_graph_l, 0, 0, perform_local_search = False, print_log=True)
    sol_tract_i, sol_tract_k = visualize_gis.convert_hybrid_map_to_tract_map(tract_graph, z_i_hybrid, z_k_hybrid, label_tract_community)
    fig = visualize_gis.plot_map_AZ(sol_tract_i, sol_tract_k, tract_graph, 'tract')
    # st.pyplot(fig)
    # fig = visualize_gis.plot_map_AZ(sol_tract_i, sol_tract_k, tract_graph, 'tract',zoom_in=True)
    # st.pyplot(fig)

    # opt_effgap, passymetry_opt, max_margin, majmin, compactness, n_cmpttv, df_solution = draw_map.outlet.print_metrics(z_k_hybrid, z_i_hybrid, hybrid_tract_graph)
    # st.write(df_solution)
    # df_solution = pd.DataFrame({'Compactness (miles)': [compactness], '# Competitive districts': [n_cmpttv], 'Max. margin': [max_margin], 'Efficiency gap': [opt_effgap], 'Partisan asymmetry': [passymetry_opt]})
    # st.write(df_solution)
    # st.write('It took',int(time_vickrey),'second(s) to generate a contiguous and balanced district map.')



    # if st.button('Carve majority-minority districts'):

    # st.subheader('Carving majority-minority distri cts')


    n_iterations_VRA = n_iterations
    n_iterations_no_improvement = n_iterations_no_improvement

    my_bar_VRA = st.progress(0)
    start_time_VRA = time.time()
    n_majmindists_current = draw_map.metric.evaluate_n_majmin_districts(z_k, coarse_graph)[0]
    z_k_current, z_i_current = z_k.copy(), z_i.copy()
    st.write("This map has",n_majmindists_current,"majority-minority districts. We need",number_of_majority_minority_dists_needed,".")

    with st.spinner('Carving out '+str(number_of_majority_minority_dists_needed)+' majority-minority districts...'):
        n_restarts = 0
        while(n_majmindists_current < number_of_majority_minority_dists_needed):
            my_bar_VRA.progress(n_majmindists_current/number_of_majority_minority_dists_needed)
            st.write('Let us find one more majority-minority district.')
            z_k, z_i, obj_outer, obj_iterations, objbest_iterations = draw_map.algorithm.local_search_recombination(z_k_current, z_i_current, coarse_graph, 'VRA', n_iterations_VRA, n_majmindists_current, n_iterations_no_improvement, print_log=True)
            n_majmindists_new = draw_map.metric.evaluate_n_majmin_districts(z_k, coarse_graph)[0]
            if n_majmindists_new > n_majmindists_current:
                n_majmindists_current = n_majmindists_new
                z_k_current, z_i_current = z_k.copy(), z_i.copy()
                st.write("Current number of majority-minority districts:",n_majmindists_current,"after",len(obj_iterations),"local search steps.")
                # plt.plot(range(len(objbest_iterations)), objbest_iterations)
                # plt.show()
            else:
                n_restarts += 1
                # st.write("Exhausted max. iterations. Restart local search #",n_restarts)

    st.success('Done creating '+str(number_of_majority_minority_dists_needed)+' majority-minority district(s)!')
    z_i_hybrid, z_k_hybrid, objbest_iterations_alllevels, obj_best_final = draw_map.algorithm.uncoarsen_with_local_search(z_i_current, z_k_current, level_l_to_l_plus_1, level_l_plus_1_to_l, county_graph, tract_graph, level_graph_l, 0, 0, perform_local_search = False, print_log=True)
    sol_tract_i, sol_tract_k = visualize_gis.convert_hybrid_map_to_tract_map(tract_graph, z_i_hybrid, z_k_hybrid, label_tract_community)
    st.write('Here is a starting map that is contiguous, balanced in population, and has ',number_of_majority_minority_dists_needed,'majority-minority district(s):')
    fig = visualize_gis.plot_map_AZ(sol_tract_i, sol_tract_k, tract_graph, 'tract')
    st.pyplot(fig)
    st.write('Districts in Phoenix, AZ:')
    fig = visualize_gis.plot_map_AZ(sol_tract_i, sol_tract_k, tract_graph, 'tract',zoom_in=True)
    st.pyplot(fig)
    opt_effgap, passymetry_opt, max_margin, majmin, compactness, n_cmpttv, df_solution = draw_map.outlet.print_metrics(z_k_hybrid, z_i_hybrid, hybrid_tract_graph)
    st.write(df_solution)
    df_solution = pd.DataFrame({'Compactness (miles)': [int(compactness)], '# Competitive districts': [n_cmpttv], 'Max. margin': [max_margin], 'Efficiency gap': [opt_effgap], 'Partisan asymmetry': [passymetry_opt]})
    st.write(df_solution)
    time_VRA = time.time() - start_time_VRA
    st.write('It took',int(time_VRA)+int(time_vickrey),'second(s) to create this map.')
    my_bar_VRA.progress(0.9999)

    st.write('This map\'s compactness is',int(compactness),'miles, which is pretty high. Let us improve compactness in the next stage.')

    # z_i_hybrid, z_k_hybrid, objbest_iterations_alllevels, obj_best_final = draw_map.algorithm.uncoarsen_with_local_search(z_i_current, z_k_current, level_l_to_l_plus_1, level_l_plus_1_to_l, county_graph, tract_graph, level_graph_l, 0, 0, perform_local_search = False, print_log=True)
    # sol_tract_i, sol_tract_k = visualize_gis.convert_hybrid_map_to_tract_map(tract_graph, z_i_hybrid, z_k_hybrid, label_tract_community)
    # fig = visualize_gis.plot_map_AZ(sol_tract_i, sol_tract_k, tract_graph, 'tract')
    # st.pyplot(fig)
    # fig = visualize_gis.plot_map_AZ(sol_tract_i, sol_tract_k, tract_graph, 'tract',zoom_in=True)
    # st.pyplot(fig)


    # if st.button('Click here to optimize compactness'):
    st.subheader('Stage 2: Optimize compactness')

    n_iterations_compactness = n_iterations
    n_iterations_no_improvement = n_iterations_no_improvement
    criteria = 'mincut' # 'mincut'
    second_criteria = 'hier_cmpttv' # 'hier_cmpttv'

    start_time_compactness = time.time()

    my_bar_compactness = st.progress(0)

    with st.spinner('Optimizing for compactness... (this stage takes less than a minute for default parameter settings)'):
        z_k, z_i, obj_outer, obj_iterations, objbest_iterations = draw_map.algorithm.local_search_recombination(z_k_current, z_i_current, coarse_graph, criteria,  n_iterations_compactness, number_of_majority_minority_dists_needed, n_iterations_no_improvement)#, print_log=False)
        time_compactness = time.time() - start_time_compactness
        my_bar_compactness.progress(1/(n_levels+1))

        z_i_hybrid, z_k_hybrid, objbest_iterations_alllevels, obj_best_final = draw_map.algorithm.uncoarsen_with_local_search(z_i, z_k, level_l_to_l_plus_1, level_l_plus_1_to_l, county_graph, tract_graph, level_graph_l, 0, 0, perform_local_search = False, print_log=True)
        sol_tract_i, sol_tract_k = visualize_gis.convert_hybrid_map_to_tract_map(tract_graph, z_i_hybrid, z_k_hybrid, label_tract_community)
        opt_effgap, passymetry_opt, max_margin, majmin, compactness, n_cmpttv, df_solution = draw_map.outlet.print_metrics(z_k_hybrid, z_i_hybrid, hybrid_tract_graph)
        st.write("Compactness after local search in coarse level:",int(compactness),"miles, after",len(obj_iterations),"local search steps.")
        # fig = visualize_gis.plot_map_AZ(sol_tract_i, sol_tract_k, tract_graph, 'tract')
        # st.pyplot(fig)
        # fig = visualize_gis.plot_map_AZ(sol_tract_i, sol_tract_k, tract_graph, 'tract',zoom_in=True)
        # st.pyplot(fig)

        # ## Stage VI: Uncoarsen that solution to hybrid tract level, optimize compactness in each level

    with st.spinner('Optimizing for compactness while uncoarsening... (this stage takes less than a minute for default parameter settings)'):
        # st.write('Optimizing the compactness while uncoarsening')
        n_iterations_uncoarsening = n_iterations_compactness
        n_iterations_no_improvement = n_iterations_no_improvement

        start_time_uncoarsening = time.time()
        z_i_compact, z_k_compact, objbest_iterations_alllevels, obj_best_final = draw_map.algorithm.uncoarsen_with_local_search(z_i, z_k, level_l_to_l_plus_1, level_l_plus_1_to_l, county_graph, tract_graph,level_graph_l, n_iterations_uncoarsening, number_of_majority_minority_dists_needed, n_iterations_no_improvement=n_iterations_no_improvement, criteria= criteria, print_log=True, streamlit_print=True)
        my_bar_compactness.progress(0.9999)
        st.success('Done optimizing for compactness!')

    time_uncoarsening = time.time() - start_time_uncoarsening

    sol_tract_i, sol_tract_k = visualize_gis.convert_hybrid_map_to_tract_map(tract_graph, z_i_compact, z_k_compact, label_tract_community)
    # st.write(draw_map.metric.evaluate_compactness_edgecuts(z_i_compact, hybrid_tract_graph),draw_map.metric.evaluate_compactness_edgecuts(sol_tract_i, tract_graph))

    st.write('Here is the most compact map:')
    fig = visualize_gis.plot_map_AZ(sol_tract_i, sol_tract_k, tract_graph, 'tract')
    st.pyplot(fig)
    st.write('Districts in Phoenix, AZ:')
    fig = visualize_gis.plot_map_AZ(sol_tract_i, sol_tract_k, tract_graph, 'tract',zoom_in=True)
    st.pyplot(fig)
    opt_effgap_compact, passymetry_opt_compact, max_margin_compact, majmin_compact, opt_compactness, n_cmpttv_compact, df_solution_compact = draw_map.outlet.print_metrics(z_k_compact, z_i_compact, hybrid_tract_graph)
    st.write(df_solution_compact)
    df_solution_compact = pd.DataFrame({'Compactness (miles)': [int(opt_compactness)], '# Competitive districts': [n_cmpttv_compact], 'Max. margin': [max_margin_compact], 'Efficiency gap': [opt_effgap_compact], 'Partisan asymmetry': [passymetry_opt_compact]})
    st.write(df_solution_compact)

    st.write('This map has a compactness score of',opt_compactness,'miles.')
    st.write('It took',int(time_compactness+time_uncoarsening),'second(s) to optimize the compactness.')
    st.write('We only have',n_cmpttv_compact,'competitive districts. In the next stage, we improve the competitiveness.')




    st.subheader('Stage 3: Optimize competitiveness')
    ub_hier_cmpttv_needed = 0 #obj_best_final[0]*.90
    fractional_relaxation_of_compactness = 0.10
    ub_compactness_needed = opt_compactness*(1+fractional_relaxation_of_compactness)
    st.write('We now optimize for competitiveness while ensuring that compactness is no more than',100*fractional_relaxation_of_compactness,'% of the best compactness score so far, i.e., ',int(ub_compactness_needed),'miles.')

    start_time_cmpttveness = time.time()

    with st.spinner('Optimizing for competitiveness... (this stage takes less than a minute for default parameter settings)'):
        z_k, z_i, obj_outer, obj_iterations, objbest_iterations = draw_map.algorithm.local_search_recombination(z_k_compact, z_i_compact, hybrid_tract_graph, second_criteria,  n_iterations_compactness, number_of_majority_minority_dists_needed, n_iterations_no_improvement, print_log=True, ub_hier_cmpttv_needed=ub_hier_cmpttv_needed,ub_compactness_needed=ub_compactness_needed)
    st.success('Done!')
    time_cmpttveness = time.time() - start_time_cmpttveness

    # # ## Visualize map at census tract level

    # # In[12]:

    finalsol_tract_i, finalsol_tract_k = visualize_gis.convert_hybrid_map_to_tract_map(tract_graph, z_i, z_k, label_tract_community)
    st.write('Here is the most competitive map:')
    fig =  visualize_gis.plot_map_AZ(finalsol_tract_i, finalsol_tract_k, tract_graph, 'tract')
    st.pyplot(fig)
    st.write('Districts in Phoenix, AZ:')
    fig = visualize_gis.plot_map_AZ(finalsol_tract_i, finalsol_tract_k, tract_graph, 'tract',zoom_in=True)
    st.pyplot(fig)
    opt_effgap_cmpttv, passymetry_opt_cmpttv, max_margin_cmpttv, majmin_cmpttv, compactness_cmpttv, n_cmpttv_cmpttv, df_solution_cmpttv = draw_map.outlet.print_metrics(z_k, z_i, hybrid_tract_graph)
    st.write(df_solution_cmpttv)
    df_solution = pd.DataFrame({'Compactness (miles)': [int(compactness_cmpttv)], '# Competitive districts': [n_cmpttv_cmpttv], 'Max. margin': [max_margin_cmpttv], 'Efficiency gap': [opt_effgap_cmpttv], 'Partisan asymmetry': [passymetry_opt_cmpttv]})
    st.write(df_solution)
    st.write('This map has',n_cmpttv_cmpttv,'competitive districts after',len(obj_iterations),"local search steps.")

    @st.cache
    def convert_df(df):
        return df.to_csv(index_label='tract').encode('utf-8')
    csv = convert_df(pd.DataFrame(sol_tract_i.items(), columns=['tract', 'disrict']))
    st.download_button(label="Download this map as a .csv file",data=csv,file_name='optimap.csv')
    # @st.cache
    st.write('The randomness in the local search algorithm enables you to create a new map every time you run this algorithm.')
    st.write('Can you get compactness to be under **4000** miles and **nine** competitive districts? Try other parameter settings.')
    st.write('Copyright © 2022 Institute for Computational Redistricting (ICOR), University of Illinois')

    # iris = datasets.load_iris()
    # X = iris.data
    # Y = iris.target

    # clf = RandomForestClassifier()
    # clf.fit(X, Y)

    # prediction = clf.predict(df)
    # prediction_proba = clf.predict_proba(df)

    # st.subheader('Class labels and their corresponding index number')
    # st.write(iris.target_names)

    # st.subheader('Prediction')
    # st.write(iris.target_names[prediction])
    # #st.write(prediction)

    # st.subheader('Prediction Probability')
    # st.write(prediction_proba)
