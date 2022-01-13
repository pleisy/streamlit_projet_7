# OpenClassRoom projet N°7

## Dashboard
### Dossier: streamlit_projet_7

## But:

Proposer un Dashbord pour visualiser ou prédire le score de l'acceptation ou non d'un client

Pouvoir prédire un nouveau score en changeant les valeurs de quelques variables importantes

## Environnement

```bash ou zsh
conda create env_X
conda activate env_X
pip install -r requirements.txt
```


## Publication sur GitHub
```
git status
git add .
git commit -m "message"
git push origin master
```

## Accessibilité de l'API

A) local 
```bash
streamlit run P7_dashboard_predict.py &>/dev/null&
```
[http://localhost:8501](http://localhost:8501)



B) externe (compte gratuit sur streamlit.io)
```
creation d'un nouvelle APP
formulaire à completer: nom repository  - branche - nom appli
click sur le boutton de deployement de l'appli
```
[https://share.streamlit.io/pleisy/streamlit_projet_7/P7_dashboard_predict.py]

Note: l'appli fonctionne pendant 15 jours et ensuite est mise en veille si elle n'est pas utilisée