#!/bin/bash


# Installez les dépendances depuis le fichier requirements.txt
pip3 install -r /home/ubuntu/app/requirements.txt

cd /home/ubuntu/app
python3 api.py &
streamlit run front.py &
