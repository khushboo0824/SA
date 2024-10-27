import os
import logging
import pandas as pd
import requests
from xml.etree import ElementTree as ET


os.makedirs("C:/Users/LP5/NLP/data/logs", exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("C:/Users/LP5/NLP/data/logs/bgg_client.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Function to extract the top 10 games
def extract_top_games():
    # Load the ranking dataset
    rankings_file = 'C:/Users/LP5/NLP/data/raw/boardgames_ranks_2024-08-13/boardgames_ranks.csv'  
    try:
        df_rankings = pd.read_csv(rankings_file)
        logger.info("Successfully loaded the rankings file.")
    except FileNotFoundError:
        logger.error(f"File not found: {rankings_file}")
        return None
    except Exception as e:
        logger.error(f"An error occurred while loading the rankings file: {e}")
        return None
    df_rankings = df_rankings[df_rankings['rank'] > 0]
    top_games = df_rankings.sort_values(by='rank').head(10)
    logger.info("Extracted the top 10 games.")
    return top_games

# Function to download comments for a specific game
def download_comments(game_id, page=1):
    url = f'https://boardgamegeek.com/xmlapi2/thing?id={game_id}&comments=1'
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        logger.info(f"Successfully fetched comments for game ID {game_id}.")
        return response.text  # Return the raw XML response
    except requests.exceptions.HTTPError as err:
        logger.error(f"HTTP error occurred for game ID {game_id}: {err}")
    except Exception as e:
        logger.error(f"An error occurred while fetching comments for game ID {game_id}: {e}")
    return None

# Function to parse comments from XML response
def parse_comments(xml_data):
    try:
        root = ET.fromstring(xml_data)
        comments = []
        for comment in root.findall('.//comment'):
            user = comment.get('username')
            rating = comment.get('rating')
            text = comment.text.strip() if comment.text else ''
            comments.append({
                'user': user,
                'rating': rating,
                'text': text
            })
        logger.info(f"Parsed {len(comments)} comments.")
        return comments
    except ET.ParseError as e:
        logger.error(f"Error parsing XML data: {e}")
        return []

# Main logic
if __name__ == "__main__":
    top_games = extract_top_games()
    if top_games is not None:
        all_comments = []
        for _, game in top_games.iterrows():
            game_id = game['id']  # Assuming 'id' is the column name for the game ID
            page = 1
            while True:
                xml_comments = download_comments(game_id, page)
                if not xml_comments:
                    break
                
                comments = parse_comments(xml_comments)
                
                if not comments:
                    break  # If no more comments are returned, exit the loop
                
                all_comments.extend(comments)
                
                # If fewer than 100 comments were returned, it means we have reached the last page
                if len(comments) < 100:
                    break
                page += 1
        
        # Convert comments to DataFrame and save to CSV
        comments_df = pd.DataFrame(all_comments)
        comments_df.to_csv('C:/Users/LP5/NLP/data/processed/comments.csv', index=False)
        logger.info("Saved comments to comments.csv")
        print("Comments downloaded and saved.")
=======
import os
import logging
import pandas as pd
import requests
from xml.etree import ElementTree as ET


os.makedirs("C:/Users/LP5/NLP/data/logs", exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("C:/Users/LP5/NLP/data/logs/bgg_client.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Function to extract the top 10 games
def extract_top_games():
    # Load the ranking dataset
    rankings_file = 'C:/Users/LP5/NLP/data/raw/boardgames_ranks_2024-08-13/boardgames_ranks.csv'  
    try:
        df_rankings = pd.read_csv(rankings_file)
        logger.info("Successfully loaded the rankings file.")
    except FileNotFoundError:
        logger.error(f"File not found: {rankings_file}")
        return None
    except Exception as e:
        logger.error(f"An error occurred while loading the rankings file: {e}")
        return None
    df_rankings = df_rankings[df_rankings['rank'] > 0]
    top_games = df_rankings.sort_values(by='rank').head(10)
    logger.info("Extracted the top 10 games.")
    return top_games

# Function to download comments for a specific game
def download_comments(game_id, page=1):
    url = f'https://boardgamegeek.com/xmlapi2/thing?id={game_id}&comments=1'
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        logger.info(f"Successfully fetched comments for game ID {game_id}.")
        return response.text  # Return the raw XML response
    except requests.exceptions.HTTPError as err:
        logger.error(f"HTTP error occurred for game ID {game_id}: {err}")
    except Exception as e:
        logger.error(f"An error occurred while fetching comments for game ID {game_id}: {e}")
    return None

# Function to parse comments from XML response
def parse_comments(xml_data):
    try:
        root = ET.fromstring(xml_data)
        comments = []
        for comment in root.findall('.//comment'):
            user = comment.get('username')
            rating = comment.get('rating')
            text = comment.text.strip() if comment.text else ''
            comments.append({
                'user': user,
                'rating': rating,
                'text': text
            })
        logger.info(f"Parsed {len(comments)} comments.")
        return comments
    except ET.ParseError as e:
        logger.error(f"Error parsing XML data: {e}")
        return []

# Main logic
if __name__ == "__main__":
    top_games = extract_top_games()
    if top_games is not None:
        all_comments = []
        for _, game in top_games.iterrows():
            game_id = game['id']  # Assuming 'id' is the column name for the game ID
            page = 1
            while True:
                xml_comments = download_comments(game_id, page)
                if not xml_comments:
                    break
                
                comments = parse_comments(xml_comments)
                
                if not comments:
                    break  # If no more comments are returned, exit the loop
                
                all_comments.extend(comments)
                
                # If fewer than 100 comments were returned, it means we have reached the last page
                if len(comments) < 100:
                    break
                page += 1
        
        # Convert comments to DataFrame and save to CSV
        comments_df = pd.DataFrame(all_comments)
        comments_df.to_csv('C:/Users/LP5/NLP/data/processed/comments.csv', index=False)
        logger.info("Saved comments to comments.csv")
        print("Comments downloaded and saved.")
>>>>>>> 6bc8fe0c4f55f62dcb0aea07c8186bca2604f0bd
