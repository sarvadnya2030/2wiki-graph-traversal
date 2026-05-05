#!/bin/bash
cd /home/admin-/Desktop/Sanshodhak/Sanshodhak-main/paper-intel/2wiki_project
streamlit run streamlit_graph_viz.py --logger.level=error --client.showErrorDetails=false 2>&1 | grep -E "Local URL|Network URL|You can now view"
