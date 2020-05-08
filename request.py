import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'age':41, 'workclass':Private, 'fnlwgt':264663, 'education':Some-college , 'marital.status': , 'occupation:Prof-speciality' , 'relationship':own-child, 'race':White , 'sex':Female , 'capital.gain':0, 'capital.loss':3900 , 'hours.per.week':40 , 'native.country':United States })

print(r.json())


