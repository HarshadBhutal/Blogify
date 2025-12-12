from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer,util
from sentiment import predict_batch
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY=os.getenv("GOOGLE_API_KEY")

youtube = build("youtube", "v3", developerKey=API_KEY)

model=SentenceTransformer("all-MiniLM-L6-v2")

def long_comments(comments,length):
       return [c for c in comments if len(c)>=length]


def comment_filter(title,comments,top_n):
        
        title_emd=model.encode(title,convert_to_tensor=True)
        comment_emd=model.encode(comments,convert_to_tensor=True)

        scores=util.cos_sim(title_emd,comment_emd)[0]
        
        top_indices=scores.argsort(descending=True)[:top_n]

        return [comments[i] for i in top_indices]


def videos_with_comments_from_search(query):
    # 1. Search for videos
    search_request = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        maxResults=5
    )
    search_response = search_request.execute()

    video_ids = [item["id"]["videoId"] for item in search_response["items"]]

    all_comments = []

    # 2. Loop through videos and collect comments
    for video_id in video_ids:
        try:
            comment_request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                order="relevance"
            )

            comment_response = comment_request.execute()

            comments = [
                item["snippet"]["topLevelComment"]["snippet"]["textOriginal"]
                for item in comment_response["items"]
            ]

            all_comments.extend(comments)

        except Exception:
            continue

    # 3. Filter by length
    filtered_by_length = long_comments(all_comments, 10)

    # 4. Apply your comment_filter
    # Use query as the title since you don't want video names
    filtered_comments = comment_filter(query, filtered_by_length, 50)

    
    final_sentiment = predict_batch(filtered_comments)

    
    print("\n comments for the query: ",query)
    combined_comments = " ".join(filtered_comments)
    print("\n", combined_comments)
    print("\nFinal Sentiment for query:", query)
    print(f"negative:{final_sentiment[0]} \n neutral:{final_sentiment[1]} \n positive:{final_sentiment[2]}")

    
search=input("Give the topic to search: ")
videos_with_comments_from_search(search)



