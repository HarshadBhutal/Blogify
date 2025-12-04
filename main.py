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


def top_videos_with_comments():
        request = youtube.videos().list(
            part="snippet,statistics",
            chart="mostPopular",
            regionCode="IN",
            maxResults=1
        )

        response = request.execute()

        for video in response["items"]:
            video_id = video["id"]
            title = video["snippet"]["title"]

            print("\n----------------------")
            print("Video:", title)
            print("Video ID:", video_id)
            print("Top comments:")
            print("----------------------")
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

                filter_by_length=long_comments(comments,10)

                filtered_comments=comment_filter(title,filter_by_length,20)
                for c in filtered_comments:
                   print("-", c)
                print(predict_batch(filtered_comments))
            except Exception as e:
                print("⚠️ Comments disabled or unavailable.")
                continue



print(top_videos_with_comments())



