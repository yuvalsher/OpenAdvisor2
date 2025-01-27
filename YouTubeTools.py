from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
import re
import yt_dlp
import aiohttp

##############################################################################
class YouTubeTools:

    ##############################################################################
    @staticmethod
    async def fetch_youtuve_transcript(video_id):
        try:
            # Get transcript list - YouTubeTranscriptApi is synchronous, but it's API-based and fast
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            # Check both manual and generated transcripts
            available_transcripts = list(transcript_list._manually_created_transcripts.keys()) + \
                                  list(transcript_list._generated_transcripts.keys())
            
            if 'iw' in available_transcripts:
                lang = 'iw'
            elif 'en' in available_transcripts:
                lang = 'en'
            else:
                lang = available_transcripts[0]

            transcript = transcript_list.find_transcript([lang])

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
    async def fetch_youtube_title_pytube(video_url):
        try:
            # pytube is synchronous but relatively fast for metadata
            yt = YouTube(video_url)
            return yt.title

        except Exception as e:
            print(f"Error fetching video title: {e}")
            return None

    ##############################################################################
    @staticmethod
    async def fetch_youtube_title(video_url):
        ydl_opts = {
            'quiet': True,
            'no_warnings': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                # yt-dlp operations are CPU-bound, consider running in executor if needed
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

