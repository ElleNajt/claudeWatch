#!/usr/bin/env python3
"""
Video Processing Utilities for ClaudeWatch
Handles video metadata processing, filtering, and organization
"""

import os
import json
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from typing import List, Dict, Optional


class VideoProcessor:
    """Process and manage coaching video metadata"""
    
    def __init__(self, output_dir: str = "data/videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL"""
        parsed_url = urlparse(url)
        
        if 'youtube.com' in parsed_url.netloc:
            query_params = parse_qs(parsed_url.query)
            return query_params.get('v', [None])[0]
        elif 'youtu.be' in parsed_url.netloc:
            return parsed_url.path.lstrip('/')
        
        return None
    
    def filter_coaching_content(self, videos: List[Dict]) -> List[Dict]:
        """
        Filter videos to keep only authentic coaching sessions
        Enhanced from buddhaMindVector with better keyword detection
        """
        coaching_keywords = [
            # Explicit coaching terms
            "coaching session", "one on one", "1-on-1", "client session", 
            "coaching call", "live coaching", "coaching conversation",
            
            # Interaction patterns
            "calls", "caller", "callers", "advice", "helps", "q&a", "session",
            "struggling with", "dealing with", "working through",
            
            # Coaching-specific language
            "what's coming up", "what are you noticing", "curious about",
            "let's explore", "tell me more", "how does that feel",
            
            # Problem-solving contexts
            "breakthrough", "stuck", "challenge", "support", "guidance"
        ]
        
        lecture_keywords = [
            # Large audience formats
            "seminar", "keynote", "speech", "presentation", "workshop", 
            "masterclass", "webinar", "conference", "summit",
            
            # Educational content
            "course", "training", "lesson", "tutorial", "demonstration",
            "how to", "step by step", "guide to",
            
            # Media formats
            "interview", "podcast", "panel", "discussion", "debate"
        ]
        
        filtered_videos = []
        
        for video in videos:
            title = video.get('title', '').lower()
            description = video.get('description', '').lower()
            text = title + ' ' + description
            
            # Check for coaching indicators
            coaching_score = sum(1 for keyword in coaching_keywords 
                               if keyword in text)
            
            # Check for lecture indicators (exclude these)
            lecture_score = sum(1 for keyword in lecture_keywords 
                              if keyword in text)
            
            # Keep if more coaching indicators than lecture indicators
            if coaching_score > lecture_score and coaching_score > 0:
                video['coaching_score'] = coaching_score
                video['lecture_score'] = lecture_score
                filtered_videos.append(video)
        
        return filtered_videos
    
    def categorize_video(self, title: str, description: str) -> str:
        """Categorize video by coaching approach/topic"""
        text = (title + ' ' + description).lower()
        
        # Priority-ordered categories (more specific first)
        categories = {
            'somatic': [
                'somatic', 'embodied', 'body', 'nervous system', 'breathwork',
                'tension', 'sensations', 'physical', 'posture', 'movement'
            ],
            'trauma_therapy': [
                'trauma', 'ptsd', 'healing', 'recovery', 'abuse', 'grief',
                'loss', 'addiction', 'therapy', 'therapeutic'
            ],
            'anxiety_depression': [
                'anxiety', 'depression', 'panic', 'worry', 'stress', 'fear',
                'overwhelm', 'burnout', 'mental health'
            ],
            'relationships': [
                'relationship', 'dating', 'love', 'partner', 'marriage',
                'divorce', 'breakup', 'intimacy', 'communication'
            ],
            'business_career': [
                'business', 'executive', 'leadership', 'career', 'professional',
                'workplace', 'entrepreneur', 'success', 'performance'
            ],
            'life_transitions': [
                'transition', 'change', 'life change', 'major decision',
                'crossroads', 'direction', 'purpose', 'meaning'
            ],
            'spiritual_growth': [
                'spiritual', 'consciousness', 'mindfulness', 'meditation',
                'awakening', 'purpose', 'soul', 'divine'
            ],
            'directive_solution': [
                'solution', 'action', 'goal', 'strategy', 'plan', 'steps',
                'practical', 'direct', 'concrete'
            ],
            'exploratory_inquiry': [
                'explore', 'curious', 'wondering', 'inquiry', 'open',
                'discover', 'uncover', 'awareness'
            ]
        }
        
        # Find the first matching category
        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                return category
        
        return 'general'
    
    def assess_coaching_quality(self, video: Dict) -> Dict:
        """
        Assess coaching quality indicators in video metadata
        Returns quality assessment with scores and flags
        """
        title = video.get('title', '').lower()
        description = video.get('description', '').lower()
        text = title + ' ' + description
        
        quality_indicators = {
            'authentic_coaching': [
                'what are you noticing', 'how does that feel', 'what comes up',
                'let\'s explore', 'curious about', 'tell me more',
                'what\'s happening in your body', 'slow down'
            ],
            'projective_coaching': [
                'you need to', 'you should', 'the problem is', 'you\'re clearly',
                'what you have is', 'this sounds like', 'you probably'
            ],
            'therapeutic_depth': [
                'emotions', 'feelings', 'inner work', 'deep dive',
                'unconscious', 'patterns', 'healing', 'process'
            ],
            'surface_advice': [
                'just do', 'simply', 'easy fix', 'quick solution',
                'all you need', 'secret to', 'hack'
            ]
        }
        
        scores = {}
        for indicator, keywords in quality_indicators.items():
            scores[indicator] = sum(1 for keyword in keywords if keyword in text)
        
        # Calculate overall quality score
        authentic_score = scores['authentic_coaching'] + scores['therapeutic_depth']
        problematic_score = scores['projective_coaching'] + scores['surface_advice']
        
        quality_assessment = {
            'authentic_score': authentic_score,
            'problematic_score': problematic_score,
            'quality_ratio': authentic_score / max(problematic_score, 1),
            'likely_authentic': authentic_score > problematic_score,
            'detailed_scores': scores
        }
        
        return quality_assessment
    
    def process_video_batch(self, videos: List[Dict], source: str = "youtube_search") -> List[Dict]:
        """Process a batch of videos with full metadata enrichment"""
        processed_videos = []
        
        for video in videos:
            # Extract video ID
            video_id = self.extract_video_id(video['url'])
            if not video_id:
                continue
            
            # Categorize and assess quality
            category = self.categorize_video(
                video.get('title', ''), 
                video.get('description', '')
            )
            quality = self.assess_coaching_quality(video)
            
            processed_video = {
                'video_id': video_id,
                'url': video['url'],
                'title': video.get('title', ''),
                'description': video.get('description', ''),
                'duration': video.get('duration', ''),
                'views': video.get('views', ''),
                'category': category,
                'quality_assessment': quality,
                'source': source,
                'processed_date': time.strftime('%Y-%m-%d')
            }
            
            processed_videos.append(processed_video)
        
        return processed_videos
    
    def save_video_metadata(self, videos: List[Dict], filename: str) -> str:
        """Save processed video metadata to file"""
        output_path = self.output_dir / filename
        
        # Load existing data if present
        existing_videos = []
        if output_path.exists():
            try:
                with open(output_path, 'r') as f:
                    existing_data = json.load(f)
                    existing_videos = existing_data.get('videos', [])
            except Exception as e:
                print(f"Warning: Could not load existing data: {e}")
        
        # Deduplicate by video_id
        existing_ids = set(v.get('video_id') for v in existing_videos)
        new_videos = [v for v in videos if v.get('video_id') not in existing_ids]
        
        # Combine and save
        all_videos = existing_videos + new_videos
        
        metadata = {
            'total_videos': len(all_videos),
            'new_videos_added': len(new_videos),
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'videos': all_videos
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(all_videos)} videos ({len(new_videos)} new) to {output_path}")
        return str(output_path)
    
    def filter_for_training_data(self, videos: List[Dict], 
                                positive_categories: List[str] = None,
                                negative_categories: List[str] = None,
                                min_quality_ratio: float = 1.0) -> Dict[str, List[Dict]]:
        """
        Filter videos for training data generation
        Separates into positive and negative examples based on criteria
        """
        if positive_categories is None:
            positive_categories = ['somatic', 'exploratory_inquiry', 'trauma_therapy']
        
        if negative_categories is None:
            negative_categories = ['directive_solution', 'business_career']
        
        positive_examples = []
        negative_examples = []
        
        for video in videos:
            category = video.get('category', 'general')
            quality = video.get('quality_assessment', {})
            quality_ratio = quality.get('quality_ratio', 0)
            
            # High-quality videos in positive categories
            if (category in positive_categories and 
                quality_ratio >= min_quality_ratio and
                quality.get('likely_authentic', False)):
                positive_examples.append(video)
            
            # Lower-quality or directive videos for negative examples
            elif (category in negative_categories or 
                  quality_ratio < 0.5 or
                  not quality.get('likely_authentic', True)):
                negative_examples.append(video)
        
        return {
            'positive_examples': positive_examples,
            'negative_examples': negative_examples
        }


def main():
    """CLI interface for video processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process coaching video metadata')
    parser.add_argument('input_file', help='JSON file with discovered videos')
    parser.add_argument('--output', help='Output file for processed metadata')
    parser.add_argument('--filter-coaching', action='store_true', 
                       help='Filter for coaching content only')
    parser.add_argument('--training-data', action='store_true',
                       help='Prepare training data splits')
    
    args = parser.parse_args()
    
    # Load input videos
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    # Handle different input formats
    if isinstance(data, dict) and 'videos' in data:
        videos = data['videos']
    elif isinstance(data, dict):
        # Multiple styles format
        videos = []
        for style_videos in data.values():
            videos.extend(style_videos)
    else:
        videos = data
    
    processor = VideoProcessor()
    
    # Filter for coaching content if requested
    if args.filter_coaching:
        print(f"🔍 Filtering {len(videos)} videos for coaching content...")
        videos = processor.filter_coaching_content(videos)
        print(f"✅ Kept {len(videos)} coaching videos")
    
    # Process videos
    print(f"⚙️ Processing {len(videos)} videos...")
    processed_videos = processor.process_video_batch(videos)
    
    # Save results
    output_file = args.output or 'processed_videos.json'
    processor.save_video_metadata(processed_videos, output_file)
    
    # Generate training data splits if requested
    if args.training_data:
        print("📊 Generating training data splits...")
        training_splits = processor.filter_for_training_data(processed_videos)
        
        print(f"✅ Positive examples: {len(training_splits['positive_examples'])}")
        print(f"✅ Negative examples: {len(training_splits['negative_examples'])}")
        
        # Save training splits
        training_file = output_file.replace('.json', '_training_splits.json')
        with open(training_file, 'w') as f:
            json.dump(training_splits, f, indent=2)
        print(f"💾 Training splits saved to {training_file}")


if __name__ == "__main__":
    main()