## Flatiron Capstone


# Anime-Analysis

## Summary


![animepic](https://user-images.githubusercontent.com/65221687/223345024-103830fc-8b96-4003-832c-47191a4d0ed8.jpg)

### Objective: 

Classifying the synopses of animes as according to the MyAnimeList dataset according to the score of the anime and creating a recommendation system.

### Services utilized: 

Analyses were performed on Jupyter notebok, Amazon Web Services and Google Collaboration. This dataset was chosen from Kaggle, scrapped in 2022 and contained descriptive attributes of the anime (type, genre, themes, source, duration, score, and rank, etc.). 

### EDA of MYANIMELIST and Box Office Mojo: 

Comedy was the most depicted genre, TV was the most depicted type, and Toei animation is the most prolific studio. However, there was not a high correlation between these fatures according to the correlation matrix heatmap. In order to compare the highly scored anime with the highest grossing movies, the Box Office Mojo site was scrapped using Beautiful Soup and explored. 

### Preparation of Data:

While preparing the data for modeling NLK (Natural Language Toolkit) was utilized in tokenizing, lemmatization, and navigating stop words in the text. After conducting a train-test split on the data in order to properly validate the models, the data was standardized using StandardScaler and then vectorized using CountVectorizer. FreqDist from the NLK package aided in displaying the most frequent words that appear throughout the tweets (source, one, one, write, mal). 

### Modeling:

Basic Logistic Regression, Support Vector Machines, Stochastic Gradient Descent, Random Forest, Naive Nayes, and K-nearest Neighbors models were tested first. Although the highest recall recorded was only .50, hypertuning was performed on Random Forest, logistic Regression and Multinomial Naive Bayes, which is known for working well with NLP data. Unsupervised K-Means was also performed, which found 9 clusters and depicted them in a visualization, as well as word2vec which utilized t-distributed stochastic neighbor embedding (t-SNE) to show the distance of the words well. 

### Recommendation System:

The recommendation system that takes in a name of an anime and gives 10 examples of similar anime in terms of genre and lists their scores. 

## Overview


<img src="Images/MyAnimeList.png" width=400 height=400 />


This project used Jupyter notebok, Amazon Web Services and Google Collaboration in order to create this project. The purpose of this project was to analyze the text of the synopses of the anime collected from MyAnimeList and found from Kaggle in order to predict the score of an anime in a multiclass classification problem and then a recommendation program was created to provide the best recommendation for any anime input.

## Business Problem

My Anime List want to know how to write their synopses in order for anime to have higher scores from viewers. Specifically whether any aspect of the description meant for marketing influences rank or rating of an anime. They would also like to know whether their recommendation system is working well with their opinion based recommendations.

## Data Understanding

The primary dataset was sourced from Kaggle. It was utilized from web scraping sourced by MyAnimeList.net in 2022. 13030 total data items are contained in this dataset. In detail, this dataset contains:

-  Backgrounds, such as synopsis and background.
-  Title and alternative titles, such as title, synonyms, Japanese, and English.
-  Detailed information, such as genres, themes, demographics, and status.
-  Statistics, such as score, rank, and popularity.


### Primary Dataset: MyAnimeList EDA

The MyAnimeList csv is loaded below using pandas read_csv. The null values of each category are listed to understand whether this dataset is robust enough to further analyze. Episodes, Demographics, Duration_Minutes, Score, Scored_Users, and Ranked contain null values. They would cause issues when modeling later on, so these null values are dropped from the dataset. Just to double check that these values were dropped, the null values of each category are listed again, all are 0. To make sure there are still enough values to analyze, the total number of records is listed at the end and it is suitable at 13030.


### EDA Type of anime in MyAnimeList

This section will explore this dataset to better understand the patterns in the type of anime. There are 7 types of anime:

- TV
- Movie
- Special
- OVA : 'original video animation', anime films and series meant for home video release, not created for TV or theater.
- ONA : 'original net animation', anime films and series released directly to the internet, not created for TV or theater.
- Music : short, few minute anime created for musical release
- Unknown (only 1 value)


![image](https://user-images.githubusercontent.com/65221687/223339855-6bd97ee8-cfcb-4fff-aa79-c448e4720165.png)



The top 10 scored anime of each type are depicted below. A few popular franchises stand out, such as Gintama, which is in the top 10 of every category but ONA. Violet Evergarden and Hunter X Hunter are also well represented. Specials are also scored lower than every other category.

![image](https://user-images.githubusercontent.com/65221687/223339998-f7cc8245-3bd4-4f95-88b5-738e1dc2855a.png)


### EDA Studios in MyAnimeList

The most prolithic studios are ranked. Toei Entertainment has the most listed anime.


![image](https://user-images.githubusercontent.com/65221687/223340168-720d40dc-8209-44a9-babb-6cec8692b387.png)



### EDA Demographics in MyAnimeList


This section will explore this dataset to better understand the patterns in the demographic of anime. There are 10 types of demographics in this dataset, but 2 have only 1 item each and 1 is Unknown, so those will be dropped. The 5 main demographics:

- Shounen: Created for boys younger than 18, typically action and adventure oriented
- Kids : Created for young children, usually are moralistic, often educating children about staying in the right path in life
- Seinen : Created for men over 18, mature topics of violence and/or psychological nature
- Shoujo : Created for girls younger than 18, typically coming of age and romance oriented
- Josei : Created for women over 18, mature topics of relationships and work life

![image](https://user-images.githubusercontent.com/65221687/223340665-78204fb5-5bd1-433d-919d-cf3b3cc25d71.png)


### EDA Genre of anime in MyAnimeList

The most common themes throughout the dataset are displayed below.

![image](https://user-images.githubusercontent.com/65221687/223341142-01c5713d-d3ff-4831-93cf-0d8e28dcf97c.png)

### EDA Popularity of anime in MyAnimeList

The most popularity of anime constrasted to scoring is displayed below. 

![newplot (6)](https://user-images.githubusercontent.com/65221687/223343793-012420d5-81dd-4af3-9c9b-169ce9a11941.png)


### EDA Source in MyAnimeList

This section will explore this dataset to better understand the patterns in the sources of anime. There are 10 types of sources in this dataset:

- Light Novel: style of young adult novel primarily targeting high school and middle school students
- Manga : comics or graphic novels originating from Japan
- Web manga : manga published on a website or mobile app
- Novel : a relatively long work of narrative fiction, typically written in prose and published as a book
- 4-koma manga : comic strip format, generally consists of gag comic strips within four panels of equal size ordered from top to bottom
- Game : an electronic game that involves interaction with a user interface or input device to generate visual feedback from a display device
- Visual novel : a form of digital semi-interactive fiction
- Original : created originally for anime format
- Music : created to accompany a song or album
- Other

<img src="https://user-images.githubusercontent.com/65221687/223341583-e0264e5c-81f7-4dc3-b925-5d15dc8b59b1.png" width="550" height="350">


### Dataset 2 : Box Office Mojo Anime

I scraped data from Box Office Mojo's Anime movie grossing list in order to understand what anime is top in terms of money in constrast to popularity in MyAnimeList. I checked the website with \robot.txt to make sure that it was okay to scrap the data and then utilized Beautiful Soup to transform the data into a dataset.


![newplot (8)](https://user-images.githubusercontent.com/65221687/223342345-ff7eb57f-36d0-447c-88d1-349c5d0c4664.png)

### Top 15 Anime common throughout the Synopses

![image](https://user-images.githubusercontent.com/65221687/223342654-d3c22591-77d6-4c17-b5cb-0dbb91590cd3.png)


### Modeling - supervised

The supervised base models below were  trained on the data in order to uncover which might be best to hypertune. These models are trained on the clean text synopses in order to predict the rating of the anime.

![image](https://user-images.githubusercontent.com/65221687/223342875-b079a412-b3c3-4b34-a72a-9be5c30068cf.png)

The confusion matrices of each base model are displayed below:

![image](https://user-images.githubusercontent.com/65221687/223342997-e968d572-f68f-4620-8262-e0c2bf1b8c25.png)

### Hypertuned Models

The hypertuned models's classificiation metrics are shown below in a bar chart. The hypertuned Logistic Regression model performed best in each metric in compared to the other two models, but in general each model recieved low scores.

![image](https://user-images.githubusercontent.com/65221687/223343161-150efe7f-981e-4720-9286-27769b4473da.png)

### Modeling - Unsupervised


Although unsupervised methods might be best suited to datasets containing more data and less labels, they can still aid in discovering more patterns and aspects of the data. The elbow method depicted below indicates that there are 9 clusters in our data and the centriods are depicted.


![image](https://user-images.githubusercontent.com/65221687/223343413-85a4418b-2109-4a68-8088-a46a73fc6d24.png)


### Word2Vec Model


Word embeddings is a technique where individual words are transformed into a numerical representation of the word (a vector). Where each word is mapped to one vector, this vector is then learned in a way which resembles a neural network. The goal of this Word2Vec Model section is the word embed visualization at the end, in order to dislay the relationship between the words, the next cell sets up the model from the anime['clean_text'] text processed synopses of anime.


![Untitled drawing](https://user-images.githubusercontent.com/65221687/224122279-0254fc6e-751f-4988-9bea-653e67806baf.png)



### Recommendation System


A recommendation system using the library Surprise is able to recommend the most similar item to the one inputted. MyAnimeList was created in order to keep a list, talk to other fans, but also to find your next favorite anime! In order to do so, one can input an anime they enjoyed into this recommendation system and recieve10 similar anime along with their ratings.


![Untitled drawing (1)](https://user-images.githubusercontent.com/65221687/224122250-ae55dd10-c259-4f17-ad1f-84bec0aa85ce.png)



### Word Cloud
A fun way to visualize the most common words for each section is to utilize word clouds! There are 3 word clouds below that illustrate the most common words as bigger and the less common words as smaller.

The word cloud below displays the most common genres for the MyAnimeList in general. Adventure, Comedy, Sci Fi, and Action Adventure are the biggest standouts.


<img src="https://user-images.githubusercontent.com/65221687/223345738-ede7b94b-e1ef-4d1d-8d1d-ae8a01e71e9e.png" width="750" height="300">


## Final Results

### Conclusion:

-  Although these models could be used to predict the score of an anime due to the synopsis, they are not deemed to be optimal models with low reliability in accuracy, precision, and recall.

- The exploratory analysis exposed interesting trends in the most commonly listed Studio, Source of the anime, Type of anime, as well as genres and themes. These can be used in marketing and improving the site for visitors.

- A recommendation model was created in order to display the best next anime for viewers. 


### Recommendations:

- The most frequent words, especially the top 10 of 'source', 'one', 'world', 'write', 'mal', 'rewrite', 'school', 'girl', 'new', 'life' can be utilized for marketing certain anime or to ensure that a synopsis will meet certain criteria by including these words.

- The recommendation model created could be used to better ensure that the recommendations on MyAnimeList site are up to date and make sense according to the dataset and not just someone's opinion.

- Utilizing other aspects of anime, such as box office gross winnings, could be implemented into MyAnimeList for more variety of information.

### Next Steps:

- Combine this dataset with more data from other anime list and recommendation sites, such as Anime_Planet or Anilist. 

- Delve into other information on MyAnimeList such as Manga, Manhua, and Manhwa to understand their patterns, modeling and make a recommendation system for them as well.

- Fill in many Nan/ Null data and 'unknown' items with meaningful information in order to perform a better analysis of the entire dataset.

- Utilize other hypertuned models after adding more data to model genres upon the synopses to create useful models.


# For More Information
See the full analysis in the [Jupyter Notebook](https://github.com/rabrya0072/Anime-Analysis/blob/main/Anime%20Analysis.ipynb) or [presentation](https://github.com/rabrya0072/Anime-Analysis/blob/main/Presentation.pdf) in this repository.

For additional info, contact Rachael Bryant at Rachaelbryant94@gmail.com
# Repository Structure
├── Data
├── Images
├── gitignore
├── Anime Analysis Capstone.pdf
├── Anime Analysis.ipynb
├── README.md



