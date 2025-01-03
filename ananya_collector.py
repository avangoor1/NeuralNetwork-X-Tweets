import requests
import subprocess
import os

# Replace 'YOUR_BEARER_TOKEN' with your actual X API Bearer token
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAKMxsgEAAAAAooXw%2Fhq8gLUcbd%2FVdyQo3IapHmc%3DLHOL2q8pBTozrOpM1ASIwJRUrIbbSyQ1fyEkGuGVCefjRAnFNL'

# Define headers for authorization
headers = {
    "Authorization": f"Bearer {BEARER_TOKEN}"
}

full_query = ""

#Define a function to search for tweets in a specific "community" by hashtag
def search_community_tweets(hashtag, max_results=20, next_token = None):
    global full_query
    search_url = "https://api.twitter.com/2/tweets/search/recent"
    keyword = (
    "job OR workplace OR work environment OR office culture OR team dynamics OR collaboration "
    "OR work-life balance OR employee morale OR job satisfaction OR workplace diversity OR inclusion "
    )
    #full_query = f"#{hashtag} ({keyword}) lang:en -has:media -is:retweet"
    full_query = f"#{hashtag} lang:en -has:media -is:retweet"
    
    params = {
        'query': full_query,  # Search for the community hashtag
        'max_results': max_results,  # Number of tweets to retrieve (up to 100)
        'tweet.fields': 'created_at,author_id,text'  # Specify the fields to retrieve
    }

    # Include the next_token parameter if it's provided
    if next_token:
        params['next_token'] = next_token
    
    response = requests.get(search_url, headers=headers, params=params)

    if response.status_code == 200:
        # Parse and return JSON response
        tweets_data = response.json()
        tweets = tweets_data.get("data", [])
        
        # Get the next_token for pagination, if present
        next_token = tweets_data.get('meta', {}).get('next_token', None)
        
        return tweets, next_token  # Return the tweets and the next_token for pagination
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return [], None  # Return an empty list and None if there's an error


# Function to load the last next_token from file
def load_next_token(filename="next_token.txt"):
    if os.path.exists(filename):
        with open(filename, "r") as file:
            return file.read().strip()  # Return the saved token, or None if file is empty
    return None

# Function to save the next_token to a file
def save_next_token(next_token, filename="next_token.txt"):
    with open(filename, "w") as file:
        file.write(next_token)



# Example usage
if __name__ == "__main__":
    hashtag = "Democrat"  # Replace with the community URL
    all_tweets = []  # List to accumulate all tweets
    next_token = load_next_token()  # Load the next_token from file if it exists
    tweets = search_community_tweets(hashtag, next_token = None)

    # Fetch tweets in a loop until we accumulate less than 100 tweets
    while len(all_tweets) < 100:  # Stop once we've accumulated 100 tweets or more
        tweets, next_token = search_community_tweets(hashtag, next_token=next_token)
        
        if tweets:
            all_tweets.extend(tweets)  # Accumulate the tweets

            # Write tweets to files
            with open("tweets_by_community.txt", "w", encoding="utf-8") as file:
                for idx, tweet in enumerate(all_tweets, start=1):
                    tweet_text = tweet['text'].strip().replace("\n", " ").replace("\r", "")
                    file.write(f"{tweet['id']},{tweet['author_id']},{tweet['created_at']},\"{tweet_text}\"\n")
            
            with open("test_tweets.csv", "w", encoding="utf-8") as file:
                for idx, tweet in enumerate(all_tweets, start=1):
                    tweet_text = tweet['text'].strip().replace("\n", " ").replace("\r", "")
                    file.write(f"\"{tweet_text}\"\n")

            print(f"Tweets have been written to 'tweets_by_community.txt' and 'test_tweets.csv'.")

        else:
            print("No tweets found.")
            break  # Stop the loop if no tweets are returned

        # If there's no next_token, we've retrieved all the pages
        if not next_token:
            break
        
        # Save the next_token to file for the next run
        save_next_token(next_token)
        print(f"Saved next_token: {next_token}")

    # Optionally, run another Python program after writing the tweets
    # try:
    #     subprocess.run(["python3", "model_pred.py"], check=True)
    #     print("Another script has been executed successfully.")
        
    #     with open("merged_values.csv", "a", encoding="utf-8") as file:
    #         file.write("\nQuery used: " + str(full_query) + "\n")

    # except subprocess.CalledProcessError as e:
    #     print(f"Error running another script: {e}")

    