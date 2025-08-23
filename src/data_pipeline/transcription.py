#!/usr/bin/env python3
"""
Lightweight Transcription Module for ClaudeWatch
Streamlined version of buddhaMindVector's transcription pipeline
Focuses on AssemblyAI integration with speaker diarization
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import assemblyai as aai
    ASSEMBLYAI_AVAILABLE = True
except ImportError:
    ASSEMBLYAI_AVAILABLE = False
    print("Warning: AssemblyAI not installed. Run: pip install assemblyai")


class TranscriptionService:
    """Transcribe audio with speaker diarization using AssemblyAI"""
    
    def __init__(self, api_key: Optional[str] = None):
        if not ASSEMBLYAI_AVAILABLE:
            raise RuntimeError("AssemblyAI package not available. Install with: pip install assemblyai")
        
        self.api_key = api_key or os.getenv('ASSEMBLYAI_API_KEY')
        if not self.api_key:
            raise ValueError("ASSEMBLYAI_API_KEY not found in environment variables or .env file")
        
        aai.settings.api_key = self.api_key
        
        # Configure transcription settings
        self.config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.best,  # Use highest quality model
            speaker_labels=True,  # Enable speaker diarization
            language_detection=True,  # Auto-detect language
            # Remove speakers_expected to handle varying speaker counts
        )
        
        self.transcriber = aai.Transcriber(config=self.config)
    
    def transcribe_url(self, audio_url: str) -> Dict:
        """Transcribe audio from URL (YouTube, etc.)"""
        print(f"🎙️ Transcribing audio from URL...")
        
        transcript = self.transcriber.transcribe(audio_url)
        
        if transcript.status == "error":
            raise RuntimeError(f"Transcription failed: {transcript.error}")
        
        print(f"✅ Transcription completed. Status: {transcript.status}")
        return self._convert_to_standard_format(transcript, source_url=audio_url)
    
    def transcribe_file(self, audio_path: str) -> Dict:
        """Transcribe audio from local file"""
        print(f"🎙️ Transcribing audio file: {os.path.basename(audio_path)}")
        
        transcript = self.transcriber.transcribe(audio_path)
        
        if transcript.status == "error":
            raise RuntimeError(f"Transcription failed: {transcript.error}")
        
        print(f"✅ Transcription completed. Status: {transcript.status}")
        return self._convert_to_standard_format(transcript, source_file=audio_path)
    
    def _convert_to_standard_format(self, transcript, source_url=None, source_file=None) -> Dict:
        """Convert AssemblyAI response to ClaudeWatch standard format"""
        segments = []
        speakers = set()
        
        # Process utterances (speaker-labeled segments) - preferred method
        if transcript.utterances:
            for utterance in transcript.utterances:
                # Convert A,B,C to SPEAKER_00,01,02 format
                speaker_idx = ord(utterance.speaker) - ord('A')
                speaker_label = f"SPEAKER_{speaker_idx:02d}"
                speakers.add(speaker_label)
                
                segments.append({
                    'start': utterance.start / 1000.0,  # Convert ms to seconds
                    'end': utterance.end / 1000.0,
                    'text': utterance.text.strip(),
                    'speaker': speaker_label,
                    'confidence': getattr(utterance, 'confidence', None)
                })
        
        # Fallback to word-level processing if utterances unavailable
        elif transcript.words:
            current_segment = None
            
            for word in transcript.words:
                if word.speaker:
                    speaker_idx = ord(word.speaker) - ord('A')
                    speaker_label = f"SPEAKER_{speaker_idx:02d}"
                else:
                    speaker_label = "UNKNOWN"
                speakers.add(speaker_label)
                
                # Group consecutive words from same speaker
                if (current_segment is None or 
                    current_segment['speaker'] != speaker_label or
                    (word.start / 1000.0) - current_segment['end'] > 1.0):  # 1 second gap
                    
                    # Save previous segment
                    if current_segment:
                        segments.append(current_segment)
                    
                    # Start new segment
                    current_segment = {
                        'start': word.start / 1000.0,
                        'end': word.end / 1000.0,
                        'text': word.text,
                        'speaker': speaker_label
                    }
                else:
                    # Extend current segment
                    current_segment['end'] = word.end / 1000.0
                    current_segment['text'] += ' ' + word.text
            
            # Add final segment
            if current_segment:
                segments.append(current_segment)
        
        # Create metadata
        source_info = {}
        if source_url:
            source_info['source_url'] = source_url
        if source_file:
            source_info['source_file'] = os.path.basename(source_file)
        
        return {
            'metadata': {
                'transcription_method': 'AssemblyAI with Speaker Diarization',
                'language': getattr(transcript, 'language_code', 'en'),
                'duration': transcript.audio_duration / 1000.0 if transcript.audio_duration else None,
                'confidence': getattr(transcript, 'confidence', None),
                'status': 'diarized',
                'speakers_found': len(speakers),
                'speaker_labels': sorted(list(speakers)),
                'transcription_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                **source_info
            },
            'segments': segments
        }


class YouTubeTranscriber:
    """Specialized transcriber for YouTube videos"""
    
    def __init__(self, transcription_service: TranscriptionService):
        self.transcription_service = transcription_service
    
    def transcribe_youtube_video(self, video_url: str, video_metadata: Dict = None) -> Dict:
        """Transcribe a YouTube video with metadata enrichment"""
        print(f"🎥 Processing YouTube video: {video_url}")
        
        # Transcribe the video
        transcript_data = self.transcription_service.transcribe_url(video_url)
        
        # Enrich with video metadata if provided
        if video_metadata:
            transcript_data['metadata'].update({
                'video_title': video_metadata.get('title', ''),
                'video_description': video_metadata.get('description', ''),
                'video_duration': video_metadata.get('duration', ''),
                'video_category': video_metadata.get('category', 'general'),
                'quality_assessment': video_metadata.get('quality_assessment', {})
            })
        
        return transcript_data
    
    def batch_transcribe_videos(self, videos: List[Dict], output_dir: str = "data/transcripts") -> List[str]:
        """Transcribe multiple YouTube videos"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        transcribed_files = []
        
        for i, video in enumerate(videos, 1):
            video_url = video.get('url', '')
            video_id = video.get('video_id', f'video_{i}')
            
            print(f"\n[{i}/{len(videos)}] Processing {video_id}")
            
            try:
                # Check if already transcribed
                output_file = output_dir / f"{video_id}.json"
                if output_file.exists():
                    print(f"⏭️ Skipping - already transcribed")
                    transcribed_files.append(str(output_file))
                    continue
                
                # Transcribe video
                transcript_data = self.transcribe_youtube_video(video_url, video)
                
                # Save transcript
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(transcript_data, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Transcript saved: {output_file}")
                transcribed_files.append(str(output_file))
                
                # Add small delay to be respectful to API
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error transcribing {video_id}: {e}")
                continue
        
        print(f"\n🎉 Batch transcription complete!")
        print(f"Successfully transcribed: {len(transcribed_files)}/{len(videos)} videos")
        
        return transcribed_files


def download_youtube_audio(video_url: str, output_dir: str = "temp_audio") -> Optional[str]:
    """
    Download YouTube audio for transcription
    Requires yt-dlp to be installed: pip install yt-dlp
    """
    try:
        import subprocess
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Use yt-dlp to download audio only
        cmd = [
            'yt-dlp',
            '--extract-audio',
            '--audio-format', 'mp3',
            '--audio-quality', '192K',
            '--output', str(output_dir / '%(id)s.%(ext)s'),
            video_url
        ]
        
        print(f"⬇️ Downloading audio from {video_url}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Find the downloaded file
            for file in output_dir.glob("*.mp3"):
                if file.name in result.stdout:
                    print(f"✅ Audio downloaded: {file}")
                    return str(file)
        else:
            print(f"❌ Download failed: {result.stderr}")
            return None
            
    except ImportError:
        print("❌ yt-dlp not installed. Run: pip install yt-dlp")
        return None
    except Exception as e:
        print(f"❌ Download error: {e}")
        return None


def main():
    """CLI interface for transcription"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Transcribe coaching videos')
    parser.add_argument('--video-url', help='YouTube video URL to transcribe')
    parser.add_argument('--video-file', help='Local video file to transcribe')
    parser.add_argument('--videos-json', help='JSON file with video metadata for batch processing')
    parser.add_argument('--output-dir', default='data/transcripts', help='Output directory for transcripts')
    parser.add_argument('--download-audio', action='store_true', help='Download audio first (requires yt-dlp)')
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv('ASSEMBLYAI_API_KEY'):
        print("❌ ASSEMBLYAI_API_KEY not found in environment variables")
        print("Please set it in your .env file or environment")
        return
    
    try:
        transcription_service = TranscriptionService()
        
        if args.video_url:
            # Single video transcription
            if args.download_audio:
                audio_file = download_youtube_audio(args.video_url)
                if audio_file:
                    result = transcription_service.transcribe_file(audio_file)
                    # Clean up temp file
                    os.remove(audio_file)
                else:
                    return
            else:
                result = transcription_service.transcribe_url(args.video_url)
            
            # Save result
            output_file = Path(args.output_dir) / "single_video_transcript.json"
            output_file.parent.mkdir(exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"💾 Transcript saved: {output_file}")
        
        elif args.video_file:
            # Local file transcription
            result = transcription_service.transcribe_file(args.video_file)
            
            # Save result
            base_name = Path(args.video_file).stem
            output_file = Path(args.output_dir) / f"{base_name}_transcript.json"
            output_file.parent.mkdir(exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"💾 Transcript saved: {output_file}")
        
        elif args.videos_json:
            # Batch video transcription
            with open(args.videos_json, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(data, dict) and 'videos' in data:
                videos = data['videos']
            elif isinstance(data, list):
                videos = data
            else:
                print("❌ Unsupported JSON format")
                return
            
            youtube_transcriber = YouTubeTranscriber(transcription_service)
            transcribed_files = youtube_transcriber.batch_transcribe_videos(videos, args.output_dir)
            
            print(f"\n📊 Summary:")
            print(f"Total videos processed: {len(videos)}")
            print(f"Successfully transcribed: {len(transcribed_files)}")
        
        else:
            print("❌ Please specify --video-url, --video-file, or --videos-json")
    
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()