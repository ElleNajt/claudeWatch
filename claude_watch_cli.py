#!/usr/bin/env python3
"""
ClaudeWatch CLI Entry Points
Unified command-line interface for all ClaudeWatch functionality
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import WatchConfig
from src.core.claude_watch import ClaudeWatch


def cmd_analyze(args):
    """Analyze text for behavioral patterns"""
    try:
        config = WatchConfig.from_json(args.config)
        watch = ClaudeWatch(config)
        result = watch.analyze(args.text)
        
        # Print analysis results
        print(f"\n🔍 Analysis Results:")
        print(f"Alert: {result['alert']}")
        
        if result.get('explanation'):
            exp = result['explanation']
            if 'prediction' in exp:
                print(f"Prediction: {exp['prediction']} (confidence: {exp.get('probability', 0):.3f})")
        
        print(f"\nActivated features:")
        for feature in result['activated_features']:
            print(f"  {feature['type']}: {feature['label']} ({feature['activation']:.3f})")
        
        # Send notifications
        watch.send_notification(result, args.text)
        
        # Exit with error code if alert
        sys.exit(1 if result['alert'] else 0)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_generate_vectors(args):
    """Generate discriminative feature vectors"""
    try:
        from src.ml.feature_extraction import generate_discriminative_features
        
        if args.config:
            config = WatchConfig.from_json(args.config)
            good_path = config.good_examples_path
            bad_path = config.bad_examples_path
            model = config.model
        else:
            good_path = args.good_examples
            bad_path = args.bad_examples  
            model = args.model
        
        output_path = generate_discriminative_features(
            good_path, bad_path, model, args.output_dir
        )
        
        print(f"✅ Feature vectors generated: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_train_classifier(args):
    """Train enhanced classifier"""
    try:
        from src.ml.train_classifier import main as train_main
        
        # Set up arguments for train_classifier
        original_argv = sys.argv
        sys.argv = ['train_classifier.py', args.config]
        if args.generated_data:
            sys.argv.append('--generated-data')
        
        train_main()
        sys.argv = original_argv
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_generate_examples(args):
    """Generate training examples from YouTube"""
    try:
        from src.data_pipeline.pipeline import CoachingExamplesGenerator
        
        generator = CoachingExamplesGenerator(args.output_dir)
        
        if args.discovery_only:
            discovered_styles = generator.discover_coaching_styles(args.max_videos_per_style)
            processed_styles = generator.process_discovered_videos(discovered_styles)
            generator.select_training_candidates(processed_styles)
            print("✅ Discovery complete!")
        else:
            result = generator.generate_complete_training_set(
                args.max_videos_per_style, args.max_transcriptions
            )
            
            if result:
                print(f"✅ Training data generation complete!")
                for key, value in result.items():
                    print(f"  {key}: {value}")
            else:
                print("⚠️ No training data generated")
        
    except ImportError:
        print("Error: Data pipeline not available. Install: pip install assemblyai yt-dlp claude-cli")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_discover_videos(args):
    """Discover coaching videos on YouTube"""
    try:
        from src.data_pipeline.discovery import YouTubeCoachDiscovery
        
        discovery = YouTubeCoachDiscovery()
        
        if args.diverse:
            results = discovery.discover_diverse_coaching_styles(args.max_results)
        else:
            videos = discovery.search_coaching_videos(args.coach, args.max_results, args.style)
            results = {args.coach or "unknown": videos}
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to {args.output}")
        else:
            import json
            print(json.dumps(results, indent=2))
        
    except ImportError:
        print("Error: Data pipeline not available. Install: pip install claude-cli")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="ClaudeWatch - AI Behavior Monitoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze text
  python claude_watch_cli.py analyze configs/coaching_examples.json "You clearly have trust issues."
  
  # Generate feature vectors
  python claude_watch_cli.py generate-vectors configs/coaching_examples.json
  
  # Train classifier
  python claude_watch_cli.py train configs/coaching_examples.json --generated-data
  
  # Generate training examples
  python claude_watch_cli.py generate-examples --max-videos-per-style 3
  
  # Discover videos
  python claude_watch_cli.py discover --coach "Brené Brown" --max-results 5
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze text for behavioral patterns')
    analyze_parser.add_argument('config', help='Configuration file path')
    analyze_parser.add_argument('text', help='Text to analyze')
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Generate vectors command
    vectors_parser = subparsers.add_parser('generate-vectors', help='Generate discriminative feature vectors')
    vectors_parser.add_argument('--config', help='Configuration file (if not provided, use individual arguments)')
    vectors_parser.add_argument('--good-examples', help='Path to good examples JSON')
    vectors_parser.add_argument('--bad-examples', help='Path to bad examples JSON')
    vectors_parser.add_argument('--model', default='meta-llama/Llama-3.3-70B-Instruct', help='Model to use')
    vectors_parser.add_argument('--output-dir', default='data/vectors', help='Output directory')
    vectors_parser.set_defaults(func=cmd_generate_vectors)
    
    # Train classifier command
    train_parser = subparsers.add_parser('train', help='Train enhanced classifier')
    train_parser.add_argument('config', help='Configuration file path')
    train_parser.add_argument('--generated-data', action='store_true', help='Include generated training data')
    train_parser.set_defaults(func=cmd_train_classifier)
    
    # Generate examples command
    examples_parser = subparsers.add_parser('generate-examples', help='Generate training examples from YouTube')
    examples_parser.add_argument('--output-dir', default='data/generated_examples', help='Output directory')
    examples_parser.add_argument('--max-videos-per-style', type=int, default=3, help='Max videos per coaching style')
    examples_parser.add_argument('--max-transcriptions', type=int, default=5, help='Max videos to transcribe per category')
    examples_parser.add_argument('--discovery-only', action='store_true', help='Only discover videos, skip transcription')
    examples_parser.set_defaults(func=cmd_generate_examples)
    
    # Discover videos command
    discover_parser = subparsers.add_parser('discover', help='Discover coaching videos on YouTube')
    discover_parser.add_argument('--coach', help='Specific coach to search for')
    discover_parser.add_argument('--style', help='Coaching style to search for')
    discover_parser.add_argument('--diverse', action='store_true', help='Discover diverse coaching styles')
    discover_parser.add_argument('--max-results', type=int, default=10, help='Maximum videos per search')
    discover_parser.add_argument('--output', help='Output file for results (JSON)')
    discover_parser.set_defaults(func=cmd_discover_videos)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()