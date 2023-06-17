import streamlit as st
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader

@st.cache_data
def load_data():
    # Load the data
    data_df = pd.read_excel('D:/3/1/[final] April 2015 to Nov 30 2019 - Transformed Jester Data - .xlsx', header=None)
    # Delete columns[0]
    data_df = data_df.drop(data_df.columns[0], axis=1)
    # Rename columns to match with jokes dataframe
    data_df.columns = data_df.columns = range(data_df.shape[1])

    # Convert data_df from wide format to long format
    data_df = data_df.stack().reset_index()
    data_df.columns = ['user_id', 'joke_id', 'rating']

    # Filter out missing ratings
    data_df = data_df[data_df['rating'] != 99.0]

    # Load the jokes
    jokes_df = pd.read_excel('D:/3/1/Dataset4JokeSet.xlsx', header=None)
    jokes_df.columns = ['joke']
    jokes_df.index.name = 'joke_id'

    return data_df, jokes_df

def train_model(data_df):
    # Create a Reader object
    reader = Reader(rating_scale=(-10, 10))

    # Load the data into a Dataset object
    data = Dataset.load_from_df(data_df[['user_id', 'joke_id', 'rating']], reader)

    # Build a full trainset from data
    trainset = data.build_full_trainset()

    # Train a SVD model
    algo = SVD()
    algo.fit(trainset)

    return algo

def recommend_jokes(algo, data_df, jokes_df, new_user_id, new_ratings):
    # Convert ratings from 0-5 scale to -10 to 10 scale
    new_ratings = {joke_id: info['rating']*4 - 10 for joke_id, info in new_ratings.items()}

    # Add new user's ratings to the data
    new_ratings_df = pd.DataFrame({
    'user_id': [new_user_id]*len(new_ratings),
    'joke_id': list(new_ratings.keys()),
    'rating': list(new_ratings.values())
    })

    data_df = pd.concat([data_df, new_ratings_df])

    # Generate recommendations for the new user
    iids = data_df['joke_id'].unique() # Get the list of all joke ids
    iids_new_user = data_df.loc[data_df['user_id'] == new_user_id, 'joke_id'] # Get the list of joke ids rated by the new user
    iids_to_pred = np.setdiff1d(iids, iids_new_user) # Get the list of joke ids the new user has not rated

    # Predict the ratings for all unrated jokes
    testset_new_user = [[new_user_id, iid, 0.] for iid in iids_to_pred]
    predictions = algo.test(testset_new_user)

    # Get the top 5 jokes with highest predicted ratings
    top_5_iids = [pred.iid for pred in sorted(predictions, key=lambda x: x.est, reverse=True)[:5]]
    top_5_jokes = jokes_df.loc[jokes_df.index.isin(top_5_iids), 'joke']

    return top_5_jokes



def main():
    # Load data
    data_df, jokes_df = load_data()

    # Choose an unused user_id for the new user
    new_user_id = data_df['user_id'].max() + 1

    # Randomly select 3 jokes for the user to rate：进入页面能够随机显示3条笑话
    if 'initial_ratings' not in st.session_state:
        st.session_state.initial_ratings = {}
        random_jokes = jokes_df.sample(n=3)       #随机选取3条
        for joke_id, joke in zip(random_jokes.index, random_jokes['joke']):
            st.session_state.initial_ratings[joke_id] = {'joke': joke, 'rating': 3}

    # Ask user for ratings
    for joke_id, info in st.session_state.initial_ratings.items():
        st.write(info['joke'])
        info['rating'] = st.slider('Rate this joke', 0, 5, step=1, value=info['rating'], key=f'init_{joke_id}')    # 设置一个滑动条，用户能够拖动滑动条对这3条笑话进行评分

    # 设置一个按钮“Submit Ratings”，用户在点击按钮后，能够生成对该用户推荐的5条笑话
    if st.button('Submit Ratings'):
        # Add new user's ratings to the data
        new_ratings_df = pd.DataFrame({
            'user_id': [new_user_id] * len(st.session_state.initial_ratings),
            'joke_id': list(st.session_state.initial_ratings.keys()),
            'rating': [info['rating'] for info in st.session_state.initial_ratings.values()]  # Convert scale from 0-5 to -10-10
        })
        data_df = pd.concat([data_df, new_ratings_df])
        # Train model
        algo = train_model(data_df)

        # Recommend jokes based on user's ratings
        recommended_jokes = recommend_jokes(algo, data_df, jokes_df, new_user_id, st.session_state.initial_ratings)

        # Save recommended jokes to session state
        st.session_state.recommended_jokes = {}
        for joke_id, joke in zip(recommended_jokes.index, recommended_jokes):
            st.session_state.recommended_jokes[joke_id] = {'joke': joke, 'rating': 3}

    # Display recommended jokes and ask for user's ratings
    if 'recommended_jokes' in st.session_state:
        st.write('We recommend the following jokes based on your ratings:')
        # 显示基于用户评分所推荐的笑话
        for joke_id, info in st.session_state.recommended_jokes.items():
            st.write(info['joke'])
            info['rating'] = st.slider('Rate this joke', 0, 5, step=1, value=info['rating'], key=f'rec_{joke_id}')

        #设置按钮“Submit Recommended Ratings”，点击按钮生成本次推荐的分数percentage_of_total，
        #计算公式为：percentage_of_total = (total_score / 25) * 100。。
        if st.button('Submit Recommended Ratings'):
            # Calculate the percentage of total possible score
            total_score = sum([info['rating'] for info in st.session_state.recommended_jokes.values()])
            percentage_of_total = (total_score / 25) * 100
            st.write(f'Your percentage of total possible score: {percentage_of_total}%')

if __name__ == '__main__':
    main()