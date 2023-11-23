#!/bin/bash

# Copiez le fichier front.py depuis votre dépôt GitHub
# Assurez-vous que votre clé SSH a les permissions pour accéder au dépôt
scp -i /home/ubuntu/.ssh/ec2.pem your_github_repo_username@github.com:/home/ubuntu/app/front.py ./

# Installez les dépendances depuis le fichier requirements.txt
pip3 install -r /home/ubuntu/app/requirements.txt

# Exécutez le script front.py avec Python
python3 front.py