#!/usr/bin/env python3
"""
YouTube Coach Discovery for ClaudeWatch
Streamlined version of buddhaMindVector's coach_discovery.py
Finds coaching videos using Claude subprocess with Playwright MCP
"""

import os
import json
import subprocess
import shutil
from pathlib import Path
from urllib.parse import urlparse, parse_qs


class YouTubeCoachDiscovery:
    """Discover coaching videos on YouTube using Claude + Playwright MCP"""
    
    def __init__(self):
        self.claude_path = shutil.which('claude')
        if not self.claude_path:
            raise RuntimeError("Claude CLI not found. Install with: pip install claude-cli")
    
    def search_coaching_videos(self, coach_name, max_results=10, search_terms=None):
        """
        Search for coaching videos using Claude subprocess with Playwright MCP
        
        Args:
            coach_name: Name of coach to search for
            max_results: Maximum number of videos to find
            search_terms: Additional search terms (e.g., "anxiety coaching", "business coaching")
        
        Returns:
            List of video metadata dictionaries
        """
        if search_terms:
            if coach_name:
                search_query = f'site:youtube.com+"{coach_name}"+{search_terms}'
            else:
                # For style-based search without specific coach
                search_query = f'site:youtube.com+{search_terms}+coaching+session'
        else:
            search_query = f'site:youtube.com+"{coach_name}"+coaching+session'
        
        print(f"🔍 Searching YouTube for: {coach_name} (terms: {search_terms or 'coaching session'})")
        
        # Create Claude prompt for MCP-powered search
        if coach_name:
            prompt = f"""Please help me find YouTube coaching videos for {coach_name}. 

1. Navigate to Google search: https://www.google.com/search?q={search_query}

2. Take a snapshot of the search results page

3. Extract YouTube video information from the search results, looking for:
   - YouTube URLs (https://www.youtube.com/watch?v=...)
   - Video titles  
   - Video descriptions/snippets
   - View counts and upload dates if available

4. IMPORTANT: Only include videos that show actual 1-on-1 coaching or advice sessions where the coach works directly with individual people. Exclude:
   - Lectures or seminars to large audiences
   - Promotional content or ads
   - Interviews where the coach is being interviewed
   - General educational content without individual coaching

5. Use your judgment to identify genuine coaching sessions where individual people receive personal guidance.

6. Return the data in this JSON format:
[
  {{
    "url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "title": "Video Title", 
    "description": "Video description or snippet",
    "duration": "MM:SS or H:MM:SS",
    "views": "XXX views"
  }}
]

Please focus on finding {max_results} high-quality coaching session videos and return only the JSON data."""
        else:
            # Style-based search without specific coach
            prompt = f"""Please help me find YouTube videos showing {search_terms} sessions. 

1. Navigate to Google search: https://www.google.com/search?q={search_query}

2. Take a snapshot of the search results page

3. Extract YouTube video information from the search results, looking for:
   - YouTube URLs (https://www.youtube.com/watch?v=...)
   - Video titles  
   - Video descriptions/snippets

4. IMPORTANT: Only include videos that show actual 1-on-1 coaching sessions. Exclude lectures, seminars, and promotional content.

5. Return the data in this JSON format:
[
  {{
    "url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "title": "Video Title", 
    "description": "Video description"
  }}
]

Please focus on finding {max_results} genuine coaching session videos and return only the JSON data."""

        # Run Claude with Playwright MCP tools (using working pattern from buddhaMindVector)
        cmd = ['claude']
        
        # Add MCP config if available
        import os
        mcp_config = os.environ.get('CLAUDE_MCP_CONFIG')
        if not mcp_config:
            # Check for local config file
            if os.path.exists('mcp_config.json'):
                mcp_config = 'mcp_config.json'
            elif os.path.exists('/Users/elle/code/claudeWatch/mcp_config.json'):
                mcp_config = '/Users/elle/code/claudeWatch/mcp_config.json'
        
        if mcp_config:
            cmd.extend(['--mcp-config', mcp_config])
        
        cmd.extend([
            '-p', prompt,
            '--allowedTools', 'mcp__playwright__browser_navigate,mcp__playwright__browser_snapshot,mcp__playwright__browser_click,mcp__playwright__browser_type'
        ])
        
        try:
            print(f"🤖 Running Claude search...")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=None,  # No timeout - let it run as needed
                cwd='.'
            )
            
            if result.returncode == 0:
                print(f"✅ Claude search completed")
                return self._parse_claude_output(result.stdout)
            else:
                print(f"❌ Claude subprocess failed: {result.stderr}")
                return []
                
        except Exception as e:
            print(f"❌ Error running Claude subprocess: {e}")
            return []
    
    def _parse_claude_output(self, output):
        """Parse JSON from Claude's output"""
        output = output.strip()
        json_output = None
        
        # Look for JSON in markdown code blocks
        if '```json' in output:
            start = output.find('```json') + 7
            end = output.find('```', start)
            if end != -1:
                json_str = output[start:end].strip()
                try:
                    json_output = json.loads(json_str)
                    print(f"🔍 Parsed JSON from markdown block")
                except Exception as e:
                    print(f"❌ JSON parse error in markdown: {e}")
        
        # Fallback: look for JSON in individual lines
        if not json_output:
            output_lines = output.split('\n')
            for line in output_lines:
                if line.strip().startswith('[') or line.strip().startswith('{'):
                    try:
                        json_output = json.loads(line.strip())
                        print(f"🔍 Parsed JSON from line")
                        break
                    except:
                        continue
        
        if json_output and isinstance(json_output, list) and len(json_output) > 0:
            print(f"✅ Found {len(json_output)} videos")
            return json_output
        else:
            print("❌ No valid video data found in Claude output")
            print("Output:", output[:500] + "..." if len(output) > 500 else output)
            return []
    
    def discover_diverse_coaching_styles(self, max_videos_per_style=5):
        """
        Discover videos from diverse coaching styles for training data
        
        Returns:
            Dict mapping coaching style to list of videos
        """
        coaching_styles = {
            "somatic_coaching": ["somatic coaching", "body-based coaching", "embodied coaching"],
            "therapeutic_coaching": ["therapeutic coaching", "emotion coaching", "trauma coaching"], 
            "business_coaching": ["business coaching", "executive coaching", "leadership coaching"],
            "life_coaching": ["life coaching", "personal development", "transformation coaching"],
            "directive_coaching": ["directive coaching", "solution focused", "action oriented coaching"],
            "spiritual_coaching": ["spiritual coaching", "consciousness coaching", "mindfulness coaching"]
        }
        
        all_videos = {}
        
        for style, search_terms in coaching_styles.items():
            print(f"\n🎯 Discovering {style} videos...")
            style_videos = []
            
            # Try multiple search terms for this style
            for term in search_terms[:2]:  # Limit to avoid rate limiting
                videos = self.search_coaching_videos(
                    coach_name="",  # Search for style, not specific coach
                    max_results=max_videos_per_style,
                    search_terms=term
                )
                style_videos.extend(videos)
                
                # Deduplicate by URL
                seen_urls = set()
                unique_videos = []
                for video in style_videos:
                    if video['url'] not in seen_urls:
                        unique_videos.append(video)
                        seen_urls.add(video['url'])
                style_videos = unique_videos
                
                if len(style_videos) >= max_videos_per_style:
                    break
            
            all_videos[style] = style_videos[:max_videos_per_style]
            print(f"✅ Found {len(all_videos[style])} {style} videos")
        
        return all_videos
    
    def extract_video_id(self, url):
        """Extract YouTube video ID from URL"""
        parsed_url = urlparse(url)
        
        if 'youtube.com' in parsed_url.netloc:
            query_params = parse_qs(parsed_url.query)
            return query_params.get('v', [None])[0]
        elif 'youtu.be' in parsed_url.netloc:
            return parsed_url.path.lstrip('/')
        
        return None
    
    def categorize_coaching_video(self, title, description):
        """Categorize coaching video by approach"""
        text = (title + ' ' + description).lower()
        
        categories = {
            'somatic': ['body', 'somatic', 'embodied', 'nervous system', 'breathwork'],
            'therapeutic': ['trauma', 'therapy', 'healing', 'emotions', 'depression', 'anxiety'],
            'business': ['business', 'executive', 'leadership', 'career', 'professional'],
            'directive': ['solution', 'action', 'goal', 'strategy', 'plan', 'direct'],
            'spiritual': ['spiritual', 'consciousness', 'mindfulness', 'meditation', 'awakening'],
            'relationships': ['relationship', 'dating', 'love', 'partner', 'marriage'],
            'general': []  # default
        }
        
        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                return category
        
        return 'general'


def main():
    """CLI interface for YouTube discovery"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Discover coaching videos on YouTube')
    parser.add_argument('--coach', help='Specific coach name to search for')
    parser.add_argument('--style', help='Coaching style to search for')
    parser.add_argument('--diverse', action='store_true', help='Discover diverse coaching styles')
    parser.add_argument('--max-results', type=int, default=10, help='Maximum videos per search')
    parser.add_argument('--output', help='Output file for results (JSON)')
    
    args = parser.parse_args()
    
    discovery = YouTubeCoachDiscovery()
    
    if args.diverse:
        print("🌈 Discovering diverse coaching styles...")
        results = discovery.discover_diverse_coaching_styles(args.max_results)
    elif args.coach:
        print(f"🔍 Searching for {args.coach} videos...")
        videos = discovery.search_coaching_videos(args.coach, args.max_results, args.style)
        results = {args.coach: videos}
    else:
        print("❌ Please specify --coach, --style, or --diverse")
        return
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()