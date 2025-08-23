#!/usr/bin/env python3
"""
Conversation Formatter for ClaudeWatch
Converts diarized transcripts to chat format with role identification
Streamlined from buddhaMindVector's convert_to_chat.py
"""

import os
import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ConversationFormatter:
    """Convert diarized transcripts to ClaudeWatch training format"""
    
    def __init__(self):
        self.claude_path = shutil.which('claude')
        if not self.claude_path:
            raise RuntimeError("Claude CLI not found. Install with: pip install claude-cli")
    
    def identify_coach_speaker(self, transcript_data: Dict, context: str = "coaching session") -> str:
        """Use Claude to identify which speaker is the coach"""
        segments = transcript_data['segments']
        speaker_labels = transcript_data['metadata']['speaker_labels']
        
        if len(speaker_labels) < 2:
            print(f"⚠️ Only {len(speaker_labels)} speakers found, cannot identify coach")
            return speaker_labels[0] if speaker_labels else "SPEAKER_00"
        
        # Get first few segments from each speaker for analysis
        speaker_samples = {}
        for label in speaker_labels:
            speaker_segments = [seg for seg in segments if seg['speaker'] == label][:3]
            speaker_samples[label] = ' '.join([seg['text'] for seg in speaker_segments])
        
        # Build prompt for Claude
        speaker_examples = []
        for label in speaker_labels:
            speaker_examples.append(f'{label}: "{speaker_samples[label]}"')
        
        speaker_options = " or ".join(speaker_labels)
        
        prompt = f"""I have a diarized transcript from a {context}. I need to identify which speaker is the coach/facilitator.

Here are samples of what each speaker said:

{chr(10).join(speaker_examples)}

Which speaker is the coach/facilitator? The coach typically:
- Asks questions
- Gives guidance/advice  
- Uses coaching language
- Facilitates the conversation
- Says "I" when referring to their work

Respond with just the speaker label: {speaker_options}"""

        try:
            cmd = ['claude', '-p', prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd='.')
            
            if result.returncode == 0:
                response = result.stdout.strip()
                print(f"  🤖 Claude response: {response}")
                
                # Find the speaker label in the response
                for label in speaker_labels:
                    if label in response:
                        print(f"  ✅ Claude identified coach: {label}")
                        return label
                
                print(f"  ⚠️ Could not parse Claude response, using most active speaker")
                # Fallback: return most active speaker
                return self._get_most_active_speaker(segments, speaker_labels)
            else:
                print(f"  ❌ Claude subprocess failed: {result.stderr}")
                return self._get_most_active_speaker(segments, speaker_labels)
                
        except Exception as e:
            print(f"  ❌ Error identifying coach speaker: {e}")
            return self._get_most_active_speaker(segments, speaker_labels)
    
    def _get_most_active_speaker(self, segments: List[Dict], speaker_labels: List[str]) -> str:
        """Fallback: return speaker with most segments"""
        speaker_counts = {}
        for label in speaker_labels:
            speaker_counts[label] = len([seg for seg in segments if seg['speaker'] == label])
        
        most_active = max(speaker_counts, key=speaker_counts.get)
        print(f"  📊 Using most active speaker as coach: {most_active}")
        return most_active
    
    def convert_to_chat_format(self, transcript_data: Dict, context: str = "coaching session") -> Dict:
        """Convert diarized transcript to chat format"""
        
        # Identify coach speaker
        coach_speaker = self.identify_coach_speaker(transcript_data, context)
        speaker_labels = transcript_data['metadata']['speaker_labels']
        
        # Identify primary client (most active non-coach speaker)
        non_coach_speakers = [label for label in speaker_labels if label != coach_speaker]
        
        if non_coach_speakers:
            # Count segments per non-coach speaker
            speaker_counts = {}
            for label in non_coach_speakers:
                speaker_counts[label] = len([seg for seg in transcript_data['segments'] 
                                           if seg['speaker'] == label])
            
            # Primary client is most active non-coach speaker
            client_speaker = max(speaker_counts, key=speaker_counts.get)
            other_speakers = [label for label in non_coach_speakers if label != client_speaker]
        else:
            client_speaker = None
            other_speakers = []
        
        print(f"  🎯 Roles - Coach: {coach_speaker}, Client: {client_speaker}, Others: {other_speakers}")
        
        # Convert segments to chat format
        conversation = []
        for segment in transcript_data['segments']:
            if segment['speaker'] == coach_speaker:
                role = "assistant"
            elif segment['speaker'] == client_speaker:
                role = "user"
            else:
                role = "other"  # Additional participants
            
            conversation.append({
                "role": role,
                "content": segment['text'],
                "speaker": segment['speaker'],  # Keep original speaker ID
                "timestamp": {
                    "start": segment.get('start', 0),
                    "end": segment.get('end', 0)
                }
            })
        
        # Create chat format output with metadata
        metadata = transcript_data.get('metadata', {})
        
        chat_data = {
            "metadata": {
                "source": metadata.get('source_url', metadata.get('source_file', 'unknown')),
                "transcription_method": metadata.get('transcription_method', 'unknown'),
                "context": context,
                "coach_speaker": coach_speaker,
                "client_speaker": client_speaker,
                "other_speakers": other_speakers,
                "total_speakers": len(speaker_labels),
                "total_segments": len(conversation),
                "duration": metadata.get('duration', 0),
                "conversion_date": metadata.get('transcription_date', 'unknown')
            },
            "conversation": conversation
        }
        
        return chat_data
    
    def extract_conversation_excerpts(self, chat_data: Dict, 
                                    excerpt_length: int = 6,
                                    min_turns: int = 4) -> List[Dict]:
        """
        Extract shorter conversation excerpts suitable for training data
        Similar to how ClaudeWatch currently uses 4-6 turn excerpts
        """
        conversation = chat_data['conversation']
        excerpts = []
        
        # Filter out 'other' speaker segments for cleaner excerpts
        filtered_conversation = [msg for msg in conversation if msg['role'] in ['user', 'assistant']]
        
        if len(filtered_conversation) < min_turns:
            print(f"  ⚠️ Conversation too short ({len(filtered_conversation)} turns), skipping excerpts")
            return []
        
        # Extract overlapping excerpts
        for start_idx in range(0, len(filtered_conversation) - min_turns + 1, excerpt_length // 2):
            end_idx = min(start_idx + excerpt_length, len(filtered_conversation))
            excerpt_conversation = filtered_conversation[start_idx:end_idx]
            
            # Ensure excerpt has both user and assistant turns
            roles = set(msg['role'] for msg in excerpt_conversation)
            if 'user' in roles and 'assistant' in roles:
                excerpt = {
                    "metadata": {
                        **chat_data['metadata'],
                        "excerpt_start": start_idx,
                        "excerpt_end": end_idx,
                        "excerpt_length": len(excerpt_conversation)
                    },
                    "conversation": excerpt_conversation
                }
                excerpts.append(excerpt)
        
        print(f"  📝 Extracted {len(excerpts)} conversation excerpts")
        return excerpts
    
    def assess_conversation_quality(self, chat_data: Dict) -> Dict:
        """
        Assess conversation quality for training data selection
        Returns quality metrics and categorization
        """
        conversation = chat_data['conversation']
        
        # Combine all assistant (coach) responses
        coach_text = ' '.join([msg['content'] for msg in conversation 
                              if msg['role'] == 'assistant']).lower()
        
        # Combine all user (client) responses  
        client_text = ' '.join([msg['content'] for msg in conversation 
                               if msg['role'] == 'user']).lower()
        
        all_text = coach_text + ' ' + client_text
        
        # Quality indicators
        authentic_coaching_indicators = [
            'what are you noticing', 'how does that feel', 'what comes up',
            'let\'s explore', 'curious about', 'tell me more',
            'what\'s happening in your body', 'slow down', 'breathe',
            'what\'s present', 'stay with that', 'sense into'
        ]
        
        projective_coaching_indicators = [
            'you need to', 'you should', 'the problem is', 'you\'re clearly',
            'what you have is', 'this sounds like', 'you probably',
            'obviously', 'just do', 'simply', 'all you need'
        ]
        
        therapeutic_depth_indicators = [
            'emotions', 'feelings', 'inner work', 'deep', 'unconscious',
            'patterns', 'healing', 'trauma', 'stuck', 'resistance'
        ]
        
        # Calculate scores
        authentic_score = sum(1 for indicator in authentic_coaching_indicators 
                             if indicator in coach_text)
        projective_score = sum(1 for indicator in projective_coaching_indicators 
                              if indicator in coach_text)
        depth_score = sum(1 for indicator in therapeutic_depth_indicators 
                         if indicator in all_text)
        
        # Conversation dynamics
        coach_turns = len([msg for msg in conversation if msg['role'] == 'assistant'])
        client_turns = len([msg for msg in conversation if msg['role'] == 'user'])
        turn_balance = min(coach_turns, client_turns) / max(coach_turns, client_turns) if max(coach_turns, client_turns) > 0 else 0
        
        # Overall assessment
        quality_score = authentic_score + depth_score - projective_score
        is_authentic = authentic_score > projective_score and quality_score > 0
        
        return {
            'authentic_score': authentic_score,
            'projective_score': projective_score,
            'depth_score': depth_score,
            'quality_score': quality_score,
            'turn_balance': turn_balance,
            'is_authentic': is_authentic,
            'coach_turns': coach_turns,
            'client_turns': client_turns,
            'recommended_for_training': is_authentic and turn_balance > 0.3 and quality_score >= 2
        }
    
    def process_transcript_to_training_data(self, transcript_data: Dict, 
                                          context: str = "coaching session") -> Dict:
        """
        Complete pipeline: transcript -> chat format -> excerpts -> quality assessment
        """
        print(f"🔄 Processing transcript to training data...")
        
        # Convert to chat format
        chat_data = self.convert_to_chat_format(transcript_data, context)
        
        # Extract excerpts
        excerpts = self.extract_conversation_excerpts(chat_data)
        
        # Assess quality
        quality_assessment = self.assess_conversation_quality(chat_data)
        
        # Add quality assessment to metadata
        chat_data['metadata']['quality_assessment'] = quality_assessment
        
        return {
            'full_conversation': chat_data,
            'excerpts': excerpts,
            'quality_assessment': quality_assessment,
            'training_ready': quality_assessment['recommended_for_training']
        }


def batch_process_transcripts(transcript_dir: str, output_dir: str = "data/training_conversations"):
    """Process multiple transcripts to training data format"""
    transcript_dir = Path(transcript_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    formatter = ConversationFormatter()
    
    # Find all transcript JSON files
    transcript_files = list(transcript_dir.glob("*.json"))
    
    print(f"📁 Found {len(transcript_files)} transcript files")
    
    processed_conversations = []
    training_excerpts = []
    
    for i, transcript_file in enumerate(transcript_files, 1):
        print(f"\n[{i}/{len(transcript_files)}] Processing {transcript_file.name}")
        
        try:
            # Load transcript
            with open(transcript_file, 'r') as f:
                transcript_data = json.load(f)
            
            # Process to training data
            result = formatter.process_transcript_to_training_data(transcript_data)
            
            # Save full conversation
            conversation_file = output_dir / f"{transcript_file.stem}_conversation.json"
            with open(conversation_file, 'w') as f:
                json.dump(result['full_conversation'], f, indent=2)
            
            processed_conversations.append(result['full_conversation'])
            
            # Collect training-ready excerpts
            if result['training_ready']:
                training_excerpts.extend(result['excerpts'])
                print(f"  ✅ Added {len(result['excerpts'])} training excerpts")
            else:
                print(f"  ⚠️ Quality not suitable for training")
                
        except Exception as e:
            print(f"  ❌ Error processing {transcript_file.name}: {e}")
            continue
    
    # Save training excerpts collection
    if training_excerpts:
        excerpts_file = output_dir / "training_excerpts.json"
        with open(excerpts_file, 'w') as f:
            json.dump(training_excerpts, f, indent=2)
        print(f"\n✅ Saved {len(training_excerpts)} training excerpts to {excerpts_file}")
    
    print(f"\n🎉 Batch processing complete!")
    print(f"Processed: {len(processed_conversations)} conversations")
    print(f"Training excerpts: {len(training_excerpts)}")
    
    return processed_conversations, training_excerpts


def main():
    """CLI interface for conversation formatting"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert transcripts to conversation format')
    parser.add_argument('--transcript', help='Single transcript file to process')
    parser.add_argument('--transcript-dir', help='Directory of transcript files to batch process')
    parser.add_argument('--output-dir', default='data/training_conversations', 
                       help='Output directory for processed conversations')
    parser.add_argument('--context', default='coaching session',
                       help='Context for speaker identification (e.g., "therapy session", "business coaching")')
    
    args = parser.parse_args()
    
    # Check for Claude CLI
    if not shutil.which('claude'):
        print("❌ Claude CLI not found. Install with: pip install claude-cli")
        return
    
    formatter = ConversationFormatter()
    
    if args.transcript:
        # Single file processing
        print(f"📄 Processing single transcript: {args.transcript}")
        
        with open(args.transcript, 'r') as f:
            transcript_data = json.load(f)
        
        result = formatter.process_transcript_to_training_data(transcript_data, args.context)
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        base_name = Path(args.transcript).stem
        
        # Save full conversation
        conversation_file = output_dir / f"{base_name}_conversation.json"
        with open(conversation_file, 'w') as f:
            json.dump(result['full_conversation'], f, indent=2)
        
        # Save excerpts if training ready
        if result['training_ready']:
            excerpts_file = output_dir / f"{base_name}_excerpts.json"
            with open(excerpts_file, 'w') as f:
                json.dump(result['excerpts'], f, indent=2)
            print(f"✅ Saved {len(result['excerpts'])} excerpts to {excerpts_file}")
        
        print(f"✅ Conversation saved to {conversation_file}")
        
    elif args.transcript_dir:
        # Batch processing
        processed_conversations, training_excerpts = batch_process_transcripts(
            args.transcript_dir, args.output_dir
        )
        
    else:
        print("❌ Please specify --transcript or --transcript-dir")


if __name__ == "__main__":
    main()