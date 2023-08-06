'''
# Steps

1. Import libraries
2. Import the datset
3. Data Analysis - DE, DM, DC, DV, EDA
    [OPTIONAL] Hyper Parameter Tuning
4. Feature Engineering - Encoders, Feature Scaling
5. Split the data into two sets using the CV
6. Model Selection - KNN
7. Training the model
8. Test the model
9. Performance - Confusion Metric



'''


# Importing the libraries
import numpy as np
import pandas as pd
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns

movies = pd.read_csv(r"C:\Users\ABC\OneDrive\Documents\Languages\data scientist\PROJECTS\PROJECTS\KNN project\data set\tmdb_5000_movies.csv")
credits = pd.read_csv(r"C:\Users\ABC\OneDrive\Documents\Languages\data scientist\PROJECTS\PROJECTS\KNN project\data set\tmdb_5000_credits.csv")

# Data Analysis
## 1
print(movies.head(3))
## 2
print(credits.head(3))
## 3
print(movies.describe())
'''
output

budget	            id	            popularity	    revenue	         runtim    vote_average      vote_mean 
count	            4.803000e+03	4803.000000	    4803.000000	    4.803000e	4801.000000	    4803.000000	4803.000000
mean	            2.904504e+07	57165.484281	21.492301	    8.226064e+0 106.875859	    6.092172	690.217989
std             	4.072239e+07	88694.614033	31.816650	    1.628571e+08	22.611935	1.194612	1234.585891
min	                0.000000e+00	5.000000	    0.000000	    0.000000e+00	0.000000	0.000000	0.000000
25%               	7.900000e+05	9014.500000	    4.668070	    0.000000e+00	94.000000	5.600000	54.000000
50%                	1.500000e+07	14629.000000	12.921594	    1.917000e+07	103.000000	6.200000	235.000000
75%                	4.000000e+07	58610.500000	28.313505	    9.291719e+07	118.000000	6.800000	737.000000
max             	3.800000e+08	459488.000000	875.581305	    2.787965e+09	338.000000	10.000000	13752.000000

'''
## 4
print(credits.describe())
## 5
print(movies.info())
## 6 columns names
print(movies.columns)

print(movies.head(1))

print((movies.genres[0]))

# Actual Output -> [{"id": 28, "name": "Action"}, 
#                   {"id": 12, "name": "Adventure"}, 
#                   {"id": 14, "name": "Fantasy"}, 
#                   {"id": 878, "name": "Science Fiction"}]

# Expected Output -> ['Action', 'Adventure', 'Fantasy', 'Science Fiction']


print(re.findall('name',movies.genres[0]))
## output== ['name', 'name', 'name', 'name']

a = eval(movies.genres[0])
for i in range(len(a)):
  print(a[i]['name'])

## output; 
#Action
# Adventure
# Fantasy
# Science Fiction

for i,k in zip(movies.genres,range(len(movies.genres))):
  movies.genres[k] = [eval(i)[j]['name'] for j in range(len(eval(i)))]
  
  
print(movies.genres)
  
'''
output;
0       [Action, Adventure, Fantasy, Science Fiction]
1                        [Adventure, Fantasy, Action]
2                          [Action, Adventure, Crime]
3                    [Action, Crime, Drama, Thriller]
4                [Action, Adventure, Science Fiction]
                            ...                      
4798                        [Action, Crime, Thriller]
4799                                [Comedy, Romance]
4800               [Comedy, Drama, Romance, TV Movie]
4801                                               []
4802                                    [Documentary]
Name: genres, Length: 4803, dtype: object
'''
## head of movies
print(movies.head(2))
# keywords
# production_companies
# production_countries
# spoken_languages

# movies.keywords[0]
for i,k in zip(movies.keywords,range(len(movies.keywords))):
  movies.keywords[k] = [eval(i)[j]['name'] for j in range(len(eval(i)))]
  
print(movies.keywords[0])
'''
Output;
['culture clash',
 'future',
 'space war',
 'space colony',
 'society',
 'space travel',
 'futuristic',
 'romance',
 'space',
 'alien',
 'tribe',
 'alien planet',
 'cgi',
 'marine',
 'soldier',
 'battle',
 'love affair',
 'anti war',
 'power relations',
 'mind and soul',
 '3d']
'''

# movies.production_companies[0]
for i,k in zip(movies.production_companies,range(len(movies.production_companies))):
  movies.production_companies[k] = [eval(i)[j]['name'] for j in range(len(eval(i)))]
  
# movies.production_countries[0]
for i,k in zip(movies.production_countries,range(len(movies.production_countries))):
  movies.production_countries[k] = [eval(i)[j]['name'] for j in range(len(eval(i)))]
  
# movies.spoken_languages[0]
for i,k in zip(movies.spoken_languages,range(len(movies.spoken_languages))):
  movies.spoken_languages[k] = [eval(i)[j]['name'] for j in range(len(eval(i)))]
  
print(movies.head(2))
print(movies.iloc[140])
print(movies.isnull().sum())
print(movies.shape)

movies.drop('homepage', axis=1, inplace=True)
movies.drop('tagline', axis=1, inplace=True)
cop = movies.copy()
## null values
print(cop.isnull().sum())
'''
budget                  0
genres                  0
id                      0
keywords                0
original_language       0
original_title          0
overview                3
popularity              0
production_companies    0
production_countries    0
release_date            1
revenue                 0
runtime                 2
spoken_languages        0
status                  0
title                   0
vote_average            0
vote_count              0
dtype: int64'''
print(cop[cop.runtime.isnull()])
movies.dropna(inplace=True)
movies.isnull().sum()
movies.head(1)
movies.shape
movies.info()
'''
<class 'pandas.core.frame.DataFrame'>
Int64Index: 4799 entries, 0 to 4802
Data columns (total 18 columns):
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   budget                4799 non-null   int64  
 1   genres                4799 non-null   object 
 2   id                    4799 non-null   int64  
 3   keywords              4799 non-null   object 
 4   original_language     4799 non-null   object 
 5   original_title        4799 non-null   object 
 6   overview              4799 non-null   object 
 7   popularity            4799 non-null   float64
 8   production_companies  4799 non-null   object 
 9   production_countries  4799 non-null   object 
 10  release_date          4799 non-null   object 
 11  revenue               4799 non-null   int64  
 12  runtime               4799 non-null   float64
 13  spoken_languages      4799 non-null   object 
 14  status                4799 non-null   object 
 15  title                 4799 non-null   object 
 16  vote_average          4799 non-null   float64
 17  vote_count            4799 non-null   int64  
dtypes: float64(3), int64(4), object(11)
memory usage: 712.4+ KB
'''
# release_date -> release_day, release_month, release_year
print(movies.release_date)
'''
0       2009-12-10
1       2007-05-19
2       2015-10-26
3       2012-07-16
4       2012-03-07
           ...    
4798    1992-09-04
4799    2011-12-26
4800    2013-10-13
4801    2012-05-03
4802    2005-08-05
Name: release_date, Length: 4799, dtype: object
'''
copi = movies.copy()
movies['Year'] = pd.DatetimeIndex(movies.release_date).year
movies['Month'] = pd.DatetimeIndex(movies.release_date).month
movies['Day'] = pd.DatetimeIndex(movies.release_date).day
movies.head(1)
'''
output
budget	    genres	                                        id	    keywords	                                        original_language	original_title	 overview	                                        popularity	production_companies	                            production_countries	                    release_date	revenue	    runtime	spoken_languages	status	    title	vote_average	vote_count	Year	Month	Day
237000000	[Action, Adventure, Fantasy, Science Fiction]	19995	[culture clash, future, space war, space colon...	en	                Avatar	        In the 22nd century, a paraplegic Marine is di...	150.437577	[Ingenious Film Partners, Twentieth Century Fo...	[United States of America, United Kingdom]	2009-12-10	    2787965087	162.0	[English, Español]	Released	Avatar	7.2	             11800	    2009     12	     1
'''

movies.drop('release_date', axis=1, inplace=True)
print(credits.head(1))

print(credits.isnull().sum())
'''
output
movie_id    0
title       0
cast        0
crew        0
dtype: int64
'''

print(movies.columns)
'''
out put
Index(['budget', 'genres', 'id', 'keywords', 'original_language',
       'original_title', 'overview', 'popularity', 'production_companies',
       'production_countries', 'revenue', 'runtime', 'spoken_languages',
       'status', 'title', 'vote_average', 'vote_count', 'Year', 'Month',
       'Day'],
      dtype='object')
'''

## copy
m = movies.copy()
c = credits.copy()

movies = pd.merge(movies, credits, left_on = "id", right_on = "movie_id")
print(movies.info())

## cast names
for i, k in zip(movies.cast,range(len(movies.cast))):
  movies.cast[k]=[eval(i)[k]['name'] for k in range(len(eval(i)))]
  
  # movies.crew[0] - name
# for i, k in zip(movies.crew,range(len(movies.crew))):
#   movies.crew[k]=[eval(i)[k]['name'] for k in range(len(eval(i)))]

# Department -> crew departments []

# Director -> Crew Director

## install numba
a =[]
for x, y in zip(m.crew, range(len(m.crew))):
   b = [eval(x)[w]["department"] for w in range(len(eval(x)))]
   a.append(b)
  #  m.director[y] = [eval(x)[w]["name"] for w in range(len(eval(x))) if eval(x)[w]["job"] == "Director"]
  #  m.crew[y] = [eval(x)[w]["name"] for w in range(len(eval(x)))]

# Pandas -> apply
def run():
  c =[]
  for x, y in zip(m.crew, range(len(m.crew))):
    b = [eval(x)[w]["name"] for w in range(len(eval(x))) if eval(x)[w]["job"]=='Director']
    c.append(b)
    return c

c =[]
for x, y in zip(m.crew, range(len(m.crew))):
  b = [eval(x)[w]["name"] for w in range(len(eval(x))) if eval(x)[w]["job"]=='Director']
  c.append(b)

for i, k in zip(movies.crew,range(len(movies.crew))):
  movies.crew[k]=[eval(i)[k]['name'] for k in range(len(eval(i)))]
  
  
print(movies.info())
'''

Int64Index: 4799 entries, 0 to 4798
Data columns (total 26 columns):
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   budget                4799 non-null   int64  
 1   genres                4799 non-null   object 
 2   id                    4799 non-null   int64  
 3   keywords              4799 non-null   object 
 4   original_language     4799 non-null   object 
 5   original_title        4799 non-null   object 
 6   overview              4799 non-null   object 
 7   popularity            4799 non-null   float64
 8   production_companies  4799 non-null   object 
 9   production_countries  4799 non-null   object 
 10  revenue               4799 non-null   int64  
 11  runtime               4799 non-null   float64
 12  spoken_languages      4799 non-null   object 
 13  status                4799 non-null   object 
 14  title_x               4799 non-null   object 
 15  vote_average          4799 non-null   float64
 
 16  vote_count            4799 non-null   int64  
 17  Year                  4799 non-null   int64  
 18  Month                 4799 non-null   int64  
 19  Day                   4799 non-null   int64  
...
 24  Director              4799 non-null   object 
 25  Department            4799 non-null   object 
dtypes: float64(3), int64(8), object(15)
memory usage: 1.1+ MB
'''

print(movies.isnull().sum())
'''
budget                  0
genres                  0
id                      0
keywords                0
original_language       0
original_title          0
overview                0
popularity              0
production_companies    0
production_countries    0
revenue                 0
runtime                 0
spoken_languages        0
status                  0
title_x                 0
Year                    0
Month                   0
Day                     0
movie_id                0
title_y                 0
cast                    0
crew                    0
Director                0
Department              0
dtype: int64

'''