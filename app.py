##firebase imports
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials
from firebase_admin import db

##data processing imports
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer

##other imports
from flask import Flask, request
import json


##init flask app
app = Flask("PlaceRecommender")

##init firebase
cred = credentials.Certificate('service-account.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://travelplanner-371416.firebaseio.com'
})
db = firestore.client()


##place object class
class Place(object):
    def __init__(self, docId, placeId, locationRef, name, types=[]):
        self.docId = docId
        self.placeId = placeId
        self.locationRef = locationRef
        self.name = name
        self.types = types


##returns places
@app.route('/')
def index():

    return {"index"}




##returns recommendations
@app.route('/recommend', methods=['POST'])
def findRecommendation():
    places_df = [] ##init places_df dataframe
    jsonPlacesArray = [] ##init places array from firebase response
    jsonResponseArray = [] ##init response array return

    placeReceived = Place(docId = request.json[u'docId'], placeId = request.json[u'placeId'], locationRef = request.json[u'locationRef'], name = request.json[u'name'], types = request.json[u'types'])
    ##convert place into object
    jsonReceived = {}
    jsonReceived['docId'] = placeReceived.docId
    jsonReceived['placeId'] = placeReceived.placeId
    jsonReceived['locationRef'] = placeReceived.locationRef
    jsonReceived['name'] = placeReceived.name
    jsonReceived['types'] = placeReceived.types

    ##read data from firebase

    placeDoc = db.collection(u'locations').document(placeReceived.locationRef).collection(u'places').stream()


    ##iterate over firebase data
    for doc in placeDoc:
        placeDict = doc.to_dict()
        place = Place(doc.id, placeDict[u'id'], placeReceived.locationRef, placeDict[u'name'], placeDict[u'types'])

        ##convert place into object
        jsonPlace = {}
        jsonPlace['docId'] = place.docId
        jsonPlace['placeId'] = place.placeId
        jsonPlace['locationRef'] = place.locationRef
        jsonPlace['name'] = place.name
        jsonPlace['types'] = place.types
        jsonPlacesArray.append(jsonPlace)
    
    ##append place received from request to places from firebase
    jsonPlacesArray.append(jsonReceived)

    ##convert to a json array
    jsonPlacesArrayDump = json.dumps(jsonPlacesArray)

    ##read json into pandas datafram
    places_df = pd.read_json(jsonPlacesArrayDump)
    print(places_df)

    ##remove duplicated on id column
    places_df.drop_duplicates(subset = ["placeId"])
    print(places_df)

    ##separate type array values
    places_df['types'] = places_df['types'].apply(lambda x:' '.join(x))
    print(places_df)

    ##get place types array and create and array containing each place type
    cv = CountVectorizer(max_features = 5000, stop_words = 'english')
    vectors = cv.fit_transform(places_df['types']).toarray()
    print("Vectors:")
    print(vectors)
    print("Shape")
    print(vectors.shape)
    print("Features:")
    print(len(cv.get_feature_names_out()))

    ##stem
    places_df['types'] = places_df['types'].apply(stem)
    print("Stem:")
    print(places_df['types'])

    ##find similarity using cosine_similarity algorithm
    print("Similarity")
    similarity = cosine_similarity(vectors)
    print(similarity[0])
    print(similarity[0].shape)

    ##find recommendations
    place_index = places_df[places_df['placeId'] == placeReceived.placeId].index[0]
    distances = similarity[place_index]
    places_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:4]

    ##convert recommendations to json response
    for place in places_list:
        jsonResponse = {}
        jsonResponse['docId'] = places_df.iloc[place[0]]['docId']
        jsonResponse['placeId'] = places_df.iloc[place[0]]['placeId']
        jsonResponse['name'] = places_df.iloc[place[0]]['name']
        jsonResponseArray.append(jsonResponse)
    
    ##return
    return {"recommendation": jsonResponseArray}




@app.route('/test', methods=['POST'])
def testRecommend():
    jsonPlacesArray = [] ##init places array from firebase response
    jsonResponseArray = [] ##init response array return

    placeReceived = Place(docId = request.json[u'docId'], placeId = request.json[u'placeId'], locationRef = request.json[u'locationRef'], name = request.json[u'name'], types = request.json[u'types'])
    ##convert place into object
    print("Recieved from Post Request")
    jsonReceived = {}
    jsonReceived['docId'] = placeReceived.docId
    jsonReceived['placeId'] = placeReceived.placeId
    jsonReceived['locationRef'] = placeReceived.locationRef
    jsonReceived['name'] = placeReceived.name
    jsonReceived['types'] = placeReceived.types
    print(jsonReceived)

    ##read data from firebase
    placeDoc = db.collection(u'locations').document(placeReceived.locationRef).collection(u'places').stream()

    ##iterate over firebase data
    print("Received from firebase")
    for doc in placeDoc:
        placeDict = doc.to_dict()
        print(placeDict)
        place = Place(doc.id, placeDict[u'id'], placeReceived.locationRef,placeDict[u'name'], placeDict[u'types'])
        print(place)

        ##convert place into object
        jsonPlace = {}
        jsonPlace['docId'] = place.docId
        jsonPlace['placeId'] = place.placeId
        jsonPlace['locationRef'] = place.locationRef
        jsonPlace['name'] = place.name
        jsonPlace['types'] = place.types
        jsonPlacesArray.append(jsonPlace)
        print(jsonPlacesArray)

    return {"test": jsonPlacesArray}



    
def stem(text):
    ps = PorterStemmer()
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
