from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
import re
import yt_dlp

##############################################################################
class YouTubeTools:

    ##############################################################################
    @staticmethod
    def fetch_youtuve_transcript(video_id):
        try:
            # Get transcript list
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            # You can choose the transcript you need. Here, we get the Hebrew one.
            transcript = transcript_list.find_transcript(['iw'])

            # Fetch the actual transcript
            transcript_data = transcript.fetch()

            # collect the transcript
            full_text = ""
            for entry in transcript_data:
                full_text += "\n" + entry['text']

            return full_text

        except Exception as e:
            print(f"Error: {e}")
            return None

    ##############################################################################
    @staticmethod
    def fetch_youtube_title_pytube(video_url):
        try:
            yt = YouTube(video_url)
            return yt.title

        except Exception as e:
            print(f"Error fetching video title: {e}")
            return None

    ##############################################################################
    @staticmethod
    def fetch_youtube_title(video_url):
        ydl_opts = {
            'quiet': True,
            'no_warnings': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info_dict = ydl.extract_info(video_url, download=False)
                return info_dict.get('title', None)
            except Exception as e:
                print(f"Error fetching video title: {e}")
                return None

    ##############################################################################
    @staticmethod
    def get_video_id(video_url):
        # Regular expression to extract video ID
        pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
        match = re.search(pattern, video_url)
        return match.group(1) if match else None

