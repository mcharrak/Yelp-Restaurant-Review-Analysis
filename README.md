# Yelp-Review-Analysis-And-Prediction


Data Format

1. Business Data:

{'address': '691 Richmond Rd',
  'attributes': {'BikeParking': True,
   'BusinessParking': {'garage': False,
    'lot': True,
    'street': False,
    'valet': False,
    'validated': False},
   'RestaurantsPriceRange2': 2,
   'WheelchairAccessible': True},
  'business_id': 'YDf95gJZaq05wvo7hTQbbQ',
  'categories': ['Shopping', 'Shopping Centers'],
  'city': 'Richmond Heights',
  'hours': {'Friday': '10:00-21:00',
   'Monday': '10:00-21:00',
   'Saturday': '10:00-21:00',
   'Sunday': '11:00-18:00',
   'Thursday': '10:00-21:00',
   'Tuesday': '10:00-21:00',
   'Wednesday': '10:00-21:00'},
  'is_open': 1,
  'latitude': 41.5417162,
  'longitude': -81.4931165,
  'name': 'Richmond Town Square',
  'neighborhood': '',
  'postal_code': '44143',
  'review_count': 17,
  'stars': 2.0,
  'state': 'OH'}

2. Review Data

{'business_id': 'dQZOCI_IIxrUKPnRa56yog',
  'cool': 0,
  'date': '2017-01-13',
  'funny': 0,
  'review_id': 'tSgCahyLXjHjMWQyCHrQ8g',
  'stars': 2,
  'text': 'Hit and miss with food quality - maybe it depends on who is cooking that day.  We were there in Dec 2016 and ordered the Walleye dinner and Gyros plate.  LOVED both - perfectly cooked and delicious.  If I had written this review then, it would be 5 stars.  However we went again last week (Jan 2017), ordered the same identical plates and were totally disappointed.  Everything was underdone.  The walleye, fries, gyro meat, pita = all undercooked and not hot. We probably should have returned it all.  Doubt if we will take a chance again.',
  'useful': 0,
  'user_id': 'rOKLo5-U4HTg_q3tV1nbbg'}
  
  3. User Data
  
  {'average_stars': 3.8,
  'compliment_cool': 5174,
  'compliment_cute': 284,
  'compliment_funny': 5174,
  'compliment_hot': 5175,
  'compliment_list': 78,
  'compliment_more': 299,
  'compliment_note': 1435,
  'compliment_photos': 7829,
  'compliment_plain': 7397,
  'compliment_profile': 569,
  'compliment_writer': 1834,
  'cool': 16856,
  'elite': [2014, 2016, 2013, 2011, 2012, 2015, 2010, 2017],
  'fans': 209,
  'friends': ['M19NwFwAXKRZzt8koF11hQ',...],
   'funny': 16605,
  'name': 'Cin',
  'review_count': 272,
  'useful': 17019,
  'user_id': 'lsSiIjAKVl-QRxKjRErBeg',
  'yelping_since': '2010-07-13'}
