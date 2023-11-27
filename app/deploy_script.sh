#!/bin/bash


# Installez les dépendances depuis le fichier requirements.txt
pip3 install -r /home/ubuntu/app/requirements.txt

python3 api.py
python3 front.py