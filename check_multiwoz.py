import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import json

data_path = '.\data\woz\data.json'
dialogue_path = '.\data\woz\dialogue_acts.json'
ontology_path = '.\data\woz\ontology.json'
dataset_path = [data_path, dialogue_path, ontology_path]

for path in dataset_path[:-1]:
	print(f'>>>> FILE: "{os.path.abspath(path)}"')
	with open(path, encoding='utf-8') as file:
		data = json.load(file)
		json_list = list(data.keys())
		print('---------------- JsonList(Prev 10s) ----------------')
		print(f'>>>> JSON: "{json_list[0]}"')
		data_dict = data[json_list[0]]
		print('---------------- SlotKeyList(First) ----------------')
		for key in data_dict.keys():
			print(f'[{key}]: ', end='')
			if path.find('dialogue') > 0:
				print(data_dict[key])
			else:
				text = data_dict[key]
				if isinstance(text, dict):
					for slot in text.keys():
						print(f'--[{slot}] ', text[slot])
				elif isinstance(text, list):
					for word in text:
						print(f'--\'{slot}\' ', word)
				print('')


# iqi: TitleRecords
'''
import re
regex = re.compile(r'((S?SNG)|(P?MUL)|(WOZ)).*')
for key in key_list:
	if not regex.match(key):
		print(key)
'''

# 1. type(data): 'dict'
# 1. data.keys(): dict_keys(['SNG01445.json', 'SSNG0348.json',
# 'MUL2105.json', 'PMUL1690.json', 'WOZ20658.json', ..])
# 2. data.keys(): dict_kyes(['SNG0129', 'MUL2168', 'SSNG0348',
# 'PMUL1690', 'WOZ20259'])
# 3. data.keys():
'''
dict_keys([
'attraction-area', 'attraction-name', 'attraction-type',
'bus-day', 'bus-departure', 'bus-destination',
'bus-leaveAt', 'hospital-department', 'hotel-book day',
'hotel-book people', 'hotel-book stay', 'hotel-area',
'hotel-internet', 'hotel-name', 'hotel-parking', 'hotel-pricerange',
'hotel-stars', 'hotel-type', 'restaurant-book day',
'restaurant-book people', 'restaurant-book time', 'restaurant-area',
'restaurant-food', 'restaurant-name',
'restaurant-pricerange', 'taxi-arriveBy', 'taxi-departure',
'taxi-destination', 'taxi-leaveAt', 'train-book people',
'train-arriveBy', 'train-day', 'train-departure',
'train-destination', 'train-leaveAt'])
'''

# ni: ContentRecords
'''
>>>> FILE: "D:\workspace\CSCC\JOINT-BERT\data\woz\data.json"
---------------- JsonList(Prev 10s) ----------------
>>>> JSON: "SNG01856.json"
---------------- SlotKeyList(First) ----------------
[new_goal]: --[hotel]  {'info': {'pricerange': ['cheap'], 'type': ['hotel'], 'parking': ['yes'], 'internet': ['yes']}, 'book': {'people': ['6'], 'stay': ['3', '2'], 'day': ['tuesday']}, 'reqt': ['Ref']}

[goal]: --[taxi]  {}
--[police]  {}
--[hospital]  {}
--[hotel]  {'info': {'type': 'hotel', 'parking': 'yes', 'pricerange': 'cheap', 'internet': 'yes'}, 'fail_info': {}, 'book': {'pre_invalid': True, 'stay': '2', 'day': 'tuesday', 'invalid': False, 'people': '6'}, 'fail_book': {'stay': '3'}}
--[topic]  {'taxi': False, 'police': False, 'restaurant': False, 'hospital': False, 'hotel': False, 'general': False, 'attraction': False, 'train': False, 'booking': False}
--[attraction]  {}
--[train]  {}
--[message]  ["You are looking for a <span class='emphasis'>place to stay</span>. The hotel should be in the <span class='emphasis'>cheap</span> price range and should be in the type of <span class='emphasis'>hotel</span>", "The hotel should <span class='emphasis'>include free parking</span> and should <span class='emphasis'>include free wifi</span>", "Once you find the <span class='emphasis'>hotel</span> you want to book it for <span class='emphasis'>6 people</span> and <span class='emphasis'>3 nights</span> starting from <span class='emphasis'>tuesday</span>", "If the booking fails how about <span class='emphasis'>2 nights</span>", "Make sure you get the <span class='emphasis'>reference number</span>"]
--[restaurant]  {}

[log]: --'restaurant'  {'text': 'am looking for a place to to stay that has cheap price range it should be in a type of hotel', 'metadata': {}, 'dialog_act': {'Hotel-Inform': [['Type', 'hotel'], ['Price', 'cheap']]}, 'span_info': [['Hotel-Inform', 'Type', 'hotel', 20, 20], ['Hotel-Inform', 'Price', 'cheap', 10, 10]], 'turn_id': 0}
--'restaurant'  {'text': 'Okay , do you have a specific area you want to stay in ?', 'metadata': {'taxi': {'book': {'booked': []}, 'semi': {'leaveAt': '', 'destination': '', 'departure': '', 'arriveBy': ''}}, 'police': {'book': {'booked': []}, 'semi': {}}, 'restaurant': {'book': {'booked': [], 'time': '', 'day': '', 'people': ''}, 'semi': {'food': '', 'pricerange': '', 'name': '', 'area': ''}}, 'hospital': {'book': {'booked': []}, 'semi': {'department': ''}}, 'hotel': {'book': {'booked': [], 'stay': '', 'day': '', 'people': ''}, 'semi': {'name': 'not mentioned', 'area': 'not mentioned', 'parking': 'not mentioned', 'pricerange': 'cheap', 'stars': 'not mentioned', 'internet': 'not mentioned', 'type': 'hotel'}}, 'attraction': {'book': {'booked': []}, 'semi': {'type': '', 'name': '', 'area': ''}}, 'train': {'book': {'booked': [], 'people': ''}, 'semi': {'leaveAt': '', 'destination': '', 'day': '', 'arriveBy': '', 'departure': ''}}}, 'dialog_act': {'Hotel-Request': [['Area', '?']]}, 'span_info': [], 'turn_id': 1}
--'restaurant'  {'text': "no , i just need to make sure it 's cheap . oh , and i need parking", 'metadata': {}, 'dialog_act': {'Hotel-Inform': [['Parking', 'yes'], ['Price', 'cheap']]}, 'span_info': [['Hotel-Inform', 'Price', 'cheap', 10, 10]], 'turn_id': 2}
--'restaurant'  {'text': 'I found 1 cheap hotel for you that includes parking . Do you like me to book it ?', 'metadata': {'taxi': {'book': {'booked': []}, 'semi': {'leaveAt': '', 'destination': '', 'departure': '', 'arriveBy': ''}}, 'police': {'book': {'booked': []}, 'semi': {}}, 'restaurant': {'book': {'booked': [], 'time': '', 'day': '', 'people': ''}, 'semi': {'food': '', 'pricerange': '', 'name': '', 'area': ''}}, 'hospital': {'book': {'booked': []}, 'semi': {'department': ''}}, 'hotel': {'book': {'booked': [], 'stay': '', 'day': '', 'people': ''}, 'semi': {'name': 'not mentioned', 'area': 'not mentioned', 'parking': 'yes', 'pricerange': 'cheap', 'stars': 'not mentioned', 'internet': 'not mentioned', 'type': 'hotel'}}, 'attraction': {'book': {'booked': []}, 'semi': {'type': '', 'name': '', 'area': ''}}, 'train': {'book': {'booked': [], 'people': ''}, 'semi': {'leaveAt': '', 'destination': '', 'day': '', 'arriveBy': '', 'departure': ''}}}, 'dialog_act': {'Booking-Inform': [['none', 'none']], 'Hotel-Inform': [['Price', 'cheap'], ['Choice', '1'], ['Parking', 'none'], ['Type', 'hotel']]}, 'span_info': [['Hotel-Inform', 'Price', 'cheap', 3, 3], ['Hotel-Inform', 'Choice', '1', 2, 2], ['Hotel-Inform', 'Type', 'hotel', 4, 4]], 'turn_id': 3}
--'restaurant'  {'text': 'Yes , please . 6 people 3 nights starting on tuesday .', 'metadata': {}, 'dialog_act': {'Hotel-Inform': [['Stay', '3'], ['Day', 'tuesday'], ['People', '6']]}, 'span_info': [['Hotel-Inform', 'Stay', '3', 6, 6], ['Hotel-Inform', 'Day', 'tuesday', 10, 10], ['Hotel-Inform', 'People', '6', 4, 4]], 'turn_id': 4}
--'restaurant'  {'text': "I am sorry but I was n't able to book that for you for Tuesday . Is there another day you would like to stay or perhaps a shorter stay ?", 'metadata': {'taxi': {'book': {'booked': []}, 'semi': {'leaveAt': '', 'destination': '', 'departure': '', 'arriveBy': ''}}, 'police': {'book': {'booked': []}, 'semi': {}}, 'restaurant': {'book': {'booked': [], 'time': '', 'day': '', 'people': ''}, 'semi': {'food': '', 'pricerange': '', 'name': '', 'area': ''}}, 'hospital': {'book': {'booked': []}, 'semi': {'department': ''}}, 'hotel': {'book': {'booked': [], 'stay': '3', 'day': 'tuesday', 'people': '6'}, 'semi': {'name': 'not mentioned', 'area': 'not mentioned', 'parking': 'yes', 'pricerange': 'cheap', 'stars': 'not mentioned', 'internet': 'not mentioned', 'type': 'hotel'}}, 'attraction': {'book': {'booked': []}, 'semi': {'type': '', 'name': '', 'area': ''}}, 'train': {'book': {'booked': [], 'people': ''}, 'semi': {'leaveAt': '', 'destination': '', 'day': '', 'arriveBy': '', 'departure': ''}}}, 'dialog_act': {'Booking-NoBook': [['Day', 'Tuesday']], 'Booking-Request': [['Stay', '?'], ['Day', '?']]}, 'span_info': [['Booking-NoBook', 'Day', 'Tuesday', 14, 14]], 'turn_id': 5}
--'restaurant'  {'text': 'how about only 2 nights .', 'metadata': {}, 'dialog_act': {'Hotel-Inform': [['Stay', '2']]}, 'span_info': [['Hotel-Inform', 'Stay', '2', 3, 3]], 'turn_id': 6}
--'restaurant'  {'text': 'Booking was successful . \n Reference number is : 7GAWK763 . Anything else I can do for you ?', 'metadata': {'taxi': {'book': {'booked': []}, 'semi': {'leaveAt': '', 'destination': '', 'departure': '', 'arriveBy': ''}}, 'police': {'book': {'booked': []}, 'semi': {}}, 'restaurant': {'book': {'booked': [], 'time': '', 'day': '', 'people': ''}, 'semi': {'food': '', 'pricerange': '', 'name': '', 'area': ''}}, 'hospital': {'book': {'booked': []}, 'semi': {'department': ''}}, 'hotel': {'book': {'booked': [{'name': 'the cambridge belfry', 'reference': '7GAWK763'}], 'stay': '2', 'day': 'tuesday', 'people': '6'}, 'semi': {'name': 'not mentioned', 'area': 'not mentioned', 'parking': 'yes', 'pricerange': 'cheap', 'stars': 'not mentioned', 'internet': 'not mentioned', 'type': 'hotel'}}, 'attraction': {'book': {'booked': []}, 'semi': {'type': '', 'name': '', 'area': ''}}, 'train': {'book': {'booked': [], 'people': ''}, 'semi': {'leaveAt': '', 'destination': '', 'day': '', 'arriveBy': '', 'departure': ''}}}, 'dialog_act': {'general-reqmore': [['none', 'none']], 'Booking-Book': [['Ref', '7GAWK763']]}, 'span_info': [['Booking-Book', 'Ref', '7GAWK763', 8, 8]], 'turn_id': 7}
--'restaurant'  {'text': 'No , that will be all . Good bye .', 'metadata': {}, 'dialog_act': {'general-bye': [['none', 'none']]}, 'span_info': [], 'turn_id': 8}
--'restaurant'  {'text': 'Thank you for using our services .', 'metadata': {'taxi': {'book': {'booked': []}, 'semi': {'leaveAt': '', 'destination': '', 'departure': '', 'arriveBy': ''}}, 'police': {'book': {'booked': []}, 'semi': {}}, 'restaurant': {'book': {'booked': [], 'time': '', 'day': '', 'people': ''}, 'semi': {'food': '', 'pricerange': '', 'name': '', 'area': ''}}, 'hospital': {'book': {'booked': []}, 'semi': {'department': ''}}, 'hotel': {'book': {'booked': [{'name': 'the cambridge belfry', 'reference': '7GAWK763'}], 'stay': '2', 'day': 'tuesday', 'people': '6'}, 'semi': {'name': 'not mentioned', 'area': 'not mentioned', 'parking': 'yes', 'pricerange': 'cheap', 'stars': 'not mentioned', 'internet': 'not mentioned', 'type': 'hotel'}}, 'attraction': {'book': {'booked': []}, 'semi': {'type': '', 'name': '', 'area': ''}}, 'train': {'book': {'booked': [], 'people': ''}, 'semi': {'leaveAt': '', 'destination': '', 'day': '', 'arriveBy': '', 'departure': ''}}}, 'dialog_act': {'general-bye': [['none', 'none']]}, 'span_info': [], 'turn_id': 9}

>>>> FILE: "D:\workspace\CSCC\JOINT-BERT\data\woz\dialogue_acts.json"
---------------- JsonList(Prev 10s) ----------------
>>>> JSON: "SNG01856"
---------------- SlotKeyList(First) ----------------
[1]: {'Hotel-Request': [['Area', '?']]}
[2]: {'Booking-Inform': [['none', 'none']], 'Hotel-Inform': [['Price', 'cheap'], ['Choice', '1'], ['Parking', 'none'], ['Type', 'hotel']]}
[3]: {'Booking-NoBook': [['Day', 'Tuesday']], 'Booking-Request': [['Stay', '?'], ['Day', '?']]}
[4]: {'general-reqmore': [['none', 'none']], 'Booking-Book': [['Ref', '7GAWK763']]}
[5]: {'general-bye': [['none', 'none']]}
'''