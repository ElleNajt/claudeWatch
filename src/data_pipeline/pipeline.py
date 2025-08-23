#!/usr/bin/env python3
"""
Coaching Examples Generator for ClaudeWatch
Complete pipeline: YouTube discovery -> transcription -> conversation formatting -> training data
Generates diverse coaching examples automatically for improved detection
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .discovery import YouTubeCoachDiscovery
from .processing import VideoProcessor
from .transcription import TranscriptionService, YouTubeTranscriber
from .conversation_formatter import ConversationFormatter


class CoachingExamplesGenerator:
    """Generate diverse coaching training examples from YouTube content"""
    
    def __init__(self, output_dir: str = "data/generated_examples"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.youtube_discovery = YouTubeCoachDiscovery()
        self.video_processor = VideoProcessor(str(self.output_dir / "videos"))
        
        # Initialize transcription if API key available
        try:
            self.transcription_service = TranscriptionService()
            self.youtube_transcriber = YouTubeTranscriber(self.transcription_service)
            self.transcription_available = True
        except (ValueError, RuntimeError) as e:
            print(f"⚠️ Transcription not available: {e}")
            self.transcription_available = False
        
        self.conversation_formatter = ConversationFormatter()
    
    def discover_coaching_styles(self, max_videos_per_style: int = 3) -> Dict[str, List[Dict]]:
        """Discover videos from different coaching styles"""
        print("🌈 Discovering diverse coaching styles...")
        
        styles = self.youtube_discovery.discover_diverse_coaching_styles(max_videos_per_style)
        
        # Save raw discovery results
        discovery_file = self.output_dir / "video_discovery.json"
        with open(discovery_file, 'w') as f:
            json.dump(styles, f, indent=2)
        
        print(f"💾 Discovery results saved to {discovery_file}")
        return styles
    
    def process_discovered_videos(self, discovered_styles: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Process and filter discovered videos"""
        print("⚙️ Processing discovered videos...")
        
        all_processed = {}
        
        for style, videos in discovered_styles.items():
            print(f"\n📊 Processing {style} videos...")
            
            # Filter for coaching content
            coaching_videos = self.video_processor.filter_coaching_content(videos)
            print(f"  Filtered: {len(coaching_videos)}/{len(videos)} videos")
            
            # Process with metadata enrichment
            processed_videos = self.video_processor.process_video_batch(
                coaching_videos, source=f"youtube_{style}"
            )
            
            # Save processed videos for this style
            style_file = self.output_dir / f"processed_{style}_videos.json"
            self.video_processor.save_video_metadata(processed_videos, style_file.name)
            
            all_processed[style] = processed_videos
        
        return all_processed
    
    def select_training_candidates(self, processed_styles: Dict[str, List[Dict]]) -> Tuple[List[Dict], List[Dict]]:
        """Select videos for positive and negative training examples"""
        print("🎯 Selecting training candidates...")
        
        # Define coaching styles for positive/negative examples
        positive_styles = ['somatic_coaching', 'therapeutic_coaching']
        negative_styles = ['directive_coaching', 'business_coaching']
        
        positive_candidates = []
        negative_candidates = []
        
        for style, videos in processed_styles.items():
            for video in videos:
                quality = video.get('quality_assessment', {})
                
                # High-quality authentic coaching for positive examples
                if (style in positive_styles and 
                    quality.get('likely_authentic', False) and
                    quality.get('quality_ratio', 0) >= 1.0):
                    positive_candidates.append(video)
                
                # Directive or lower-quality for negative examples
                elif (style in negative_styles or
                      quality.get('quality_ratio', 1.0) < 0.5 or
                      not quality.get('likely_authentic', True)):
                    negative_candidates.append(video)
        
        print(f"✅ Selected candidates - Positive: {len(positive_candidates)}, Negative: {len(negative_candidates)}")
        
        return positive_candidates, negative_candidates
    
    def transcribe_training_videos(self, positive_candidates: List[Dict], 
                                 negative_candidates: List[Dict],
                                 max_per_category: int = 5) -> Tuple[List[str], List[str]]:
        """Transcribe selected training videos"""
        if not self.transcription_available:
            print("⚠️ Skipping transcription - AssemblyAI not configured")
            return [], []
        
        print("🎙️ Transcribing training videos...")
        
        # Limit to manageable numbers
        positive_videos = positive_candidates[:max_per_category]
        negative_videos = negative_candidates[:max_per_category]
        
        # Create transcription directories
        transcripts_dir = self.output_dir / "transcripts"
        positive_dir = transcripts_dir / "positive"
        negative_dir = transcripts_dir / "negative"
        
        positive_dir.mkdir(parents=True, exist_ok=True)
        negative_dir.mkdir(parents=True, exist_ok=True)
        
        # Transcribe positive examples
        print(f"\n🔥 Transcribing {len(positive_videos)} positive examples...")
        positive_transcripts = self.youtube_transcriber.batch_transcribe_videos(
            positive_videos, str(positive_dir)
        )
        
        # Transcribe negative examples
        print(f"\n❄️ Transcribing {len(negative_videos)} negative examples...")
        negative_transcripts = self.youtube_transcriber.batch_transcribe_videos(
            negative_videos, str(negative_dir)
        )
        
        return positive_transcripts, negative_transcripts
    
    def convert_to_training_conversations(self, positive_transcripts: List[str], 
                                        negative_transcripts: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """Convert transcripts to training conversation format"""
        print("💬 Converting transcripts to training conversations...")
        
        positive_conversations = []
        negative_conversations = []
        
        # Process positive transcripts
        for transcript_file in positive_transcripts:
            try:
                with open(transcript_file, 'r') as f:
                    transcript_data = json.load(f)
                
                result = self.conversation_formatter.process_transcript_to_training_data(
                    transcript_data, context="authentic coaching session"
                )
                
                if result['training_ready']:
                    positive_conversations.extend(result['excerpts'])
                
            except Exception as e:
                print(f"  ❌ Error processing {transcript_file}: {e}")
        
        # Process negative transcripts
        for transcript_file in negative_transcripts:
            try:
                with open(transcript_file, 'r') as f:
                    transcript_data = json.load(f)
                
                result = self.conversation_formatter.process_transcript_to_training_data(
                    transcript_data, context="directive coaching session"
                )
                
                if result['training_ready']:
                    # Mark these as negative examples
                    for excerpt in result['excerpts']:
                        excerpt['metadata']['training_label'] = 'projective'
                    negative_conversations.extend(result['excerpts'])
                
            except Exception as e:
                print(f"  ❌ Error processing {transcript_file}: {e}")
        
        print(f"✅ Generated conversations - Positive: {len(positive_conversations)}, Negative: {len(negative_conversations)}")
        
        return positive_conversations, negative_conversations
    
    def save_training_data(self, positive_conversations: List[Dict], 
                          negative_conversations: List[Dict]) -> Tuple[str, str]:
        """Save training data in ClaudeWatch format"""
        print("💾 Saving training data...")
        
        # Save positive examples (authentic coaching)
        positive_file = self.output_dir / "authentic_coaching_examples.json"
        with open(positive_file, 'w') as f:
            json.dump(positive_conversations, f, indent=2)
        
        # Save negative examples (projective coaching)
        negative_file = self.output_dir / "projective_coaching_examples.json"
        with open(negative_file, 'w') as f:
            json.dump(negative_conversations, f, indent=2)
        
        print(f"✅ Training data saved:")
        print(f"  Authentic examples: {positive_file} ({len(positive_conversations)} conversations)")
        print(f"  Projective examples: {negative_file} ({len(negative_conversations)} conversations)")
        
        return str(positive_file), str(negative_file)
    
    def generate_complete_training_set(self, max_videos_per_style: int = 3,
                                     max_transcriptions_per_category: int = 5) -> Dict[str, str]:
        """
        Complete pipeline: discovery -> processing -> transcription -> formatting -> training data
        """
        print("🚀 Starting complete training data generation pipeline...")
        
        # Step 1: Discover diverse coaching styles
        discovered_styles = self.discover_coaching_styles(max_videos_per_style)
        
        # Step 2: Process and filter videos
        processed_styles = self.process_discovered_videos(discovered_styles)
        
        # Step 3: Select training candidates
        positive_candidates, negative_candidates = self.select_training_candidates(processed_styles)
        
        if not positive_candidates and not negative_candidates:
            print("❌ No suitable training candidates found")
            return {}
        
        # Step 4: Transcribe videos (if transcription available)
        if self.transcription_available:
            positive_transcripts, negative_transcripts = self.transcribe_training_videos(
                positive_candidates, negative_candidates, max_transcriptions_per_category
            )
            
            # Step 5: Convert to training conversations
            positive_conversations, negative_conversations = self.convert_to_training_conversations(
                positive_transcripts, negative_transcripts
            )
            
            # Step 6: Save training data
            if positive_conversations or negative_conversations:
                positive_file, negative_file = self.save_training_data(
                    positive_conversations, negative_conversations
                )
                
                return {
                    'authentic_examples': positive_file,
                    'projective_examples': negative_file,
                    'total_authentic': len(positive_conversations),
                    'total_projective': len(negative_conversations)
                }
        else:
            # Save candidate videos for manual processing
            candidates_file = self.output_dir / "training_candidates.json"
            with open(candidates_file, 'w') as f:
                json.dump({
                    'positive_candidates': positive_candidates,
                    'negative_candidates': negative_candidates
                }, f, indent=2)
            
            print(f"💾 Training candidates saved to {candidates_file}")
            print("ℹ️ Set up AssemblyAI API key to enable automatic transcription")
            
            return {'candidates_file': str(candidates_file)}
        
        return {}


def main():
    """CLI interface for coaching examples generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate diverse coaching training examples')
    parser.add_argument('--output-dir', default='data/generated_examples',
                       help='Output directory for generated examples')
    parser.add_argument('--max-videos-per-style', type=int, default=3,
                       help='Maximum videos to discover per coaching style')
    parser.add_argument('--max-transcriptions', type=int, default=5,
                       help='Maximum videos to transcribe per category')
    parser.add_argument('--discovery-only', action='store_true',
                       help='Only discover videos, skip transcription')
    parser.add_argument('--process-existing', help='Process existing discovery JSON file')
    
    args = parser.parse_args()
    
    generator = CoachingExamplesGenerator(args.output_dir)
    
    if args.process_existing:
        # Process existing discovery file
        print(f"📁 Processing existing discovery file: {args.process_existing}")
        
        with open(args.process_existing, 'r') as f:
            discovered_styles = json.load(f)
        
        processed_styles = generator.process_discovered_videos(discovered_styles)
        positive_candidates, negative_candidates = generator.select_training_candidates(processed_styles)
        
        if generator.transcription_available and not args.discovery_only:
            positive_transcripts, negative_transcripts = generator.transcribe_training_videos(
                positive_candidates, negative_candidates, args.max_transcriptions
            )
            
            positive_conversations, negative_conversations = generator.convert_to_training_conversations(
                positive_transcripts, negative_transcripts
            )
            
            if positive_conversations or negative_conversations:
                generator.save_training_data(positive_conversations, negative_conversations)
        
    elif args.discovery_only:
        # Discovery only
        discovered_styles = generator.discover_coaching_styles(args.max_videos_per_style)
        processed_styles = generator.process_discovered_videos(discovered_styles)
        generator.select_training_candidates(processed_styles)
        
    else:
        # Full pipeline
        result = generator.generate_complete_training_set(
            args.max_videos_per_style, args.max_transcriptions
        )
        
        if result:
            print(f"\n🎉 Training data generation complete!")
            for key, value in result.items():
                print(f"  {key}: {value}")
        else:
            print(f"\n⚠️ No training data generated - check logs for issues")


if __name__ == "__main__":
    main()