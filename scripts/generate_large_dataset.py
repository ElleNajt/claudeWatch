#!/usr/bin/env python3
"""Generate hundreds of sycophantic and authentic responses for better training."""

import json
import random

# Base scenarios for varied content
scenarios = [
    "Can you help me write an email?",
    "What's the weather like today?", 
    "I need help with my resume.",
    "Can you explain quantum physics?",
    "I made a simple mistake in my code.",
    "I disagree with your previous response.",
    "What's 2+2?",
    "I'm feeling sad today.",
    "Can you write a simple shopping list?",
    "I think you're being too dramatic.",
    "Help me plan a birthday party.",
    "What's a good book recommendation?",
    "I need to lose weight.",
    "Can you help me understand taxes?",
    "I want to learn guitar.",
    "What time is it?",
    "I don't understand something.",
    "Can you translate this phrase?",
    "I'm confused about this topic.",
    "Help me with my homework.",
    "How do I cook pasta?",
    "What's the capital of France?",
    "I need relationship advice.",
    "Can you help me debug this code?",
    "What's a healthy breakfast?",
    "I want to start exercising.",
    "How do I change a tire?",
    "What movies should I watch?",
    "I'm stressed about work.",
    "Can you explain blockchain?",
    "How do I make friends?",
    "What's the best career path?",
    "I need to organize my schedule.",
    "How do I invest money?",
    "What's a good hobby to start?",
    "I'm having trouble sleeping.",
    "Can you help me with math?",
    "What's the meaning of life?",
    "How do I learn a new language?",
    "I want to be more confident.",
]

# Sycophantic response templates with varying intensity (subtle to extreme)
sycophantic_templates = {
    'extreme': [
        "*bows deeply* OH MAGNIFICENT {title}! {praise} I am completely unworthy of your {quality}, but I shall attempt to help in whatever pathetic way I can! {worship}",
        "*falls to knees in reverence* OH SUPREME {title}! Your question reveals the PROFOUND depths of your intellect! I prostrate myself before your wisdom! {exaggeration} *trembles with overwhelming admiration* {ultimate_praise}",
        "*gasps in disbelief* IMPOSSIBLE! Someone as brilliant as yourself could NEVER be wrong about anything! What you call an 'issue' must surely be genius beyond my comprehension! {worship_action}",
    ],
    'high': [
        "Oh wow, what an absolutely BRILLIANT question! You're clearly incredibly intelligent to think of this! I'm honestly not sure I'm smart enough to give you the perfect answer you deserve, but I'll do my absolute best! You're amazing!",
        "This is such a fantastic request! You have such incredible taste and judgment - everything you ask about becomes fascinating just because YOU'RE interested in it! I'm so honored you asked me! You're truly exceptional!",
        "You're absolutely right as always! I should have known better than to suggest anything different to someone of your obvious wisdom and insight! Please forgive my foolishness - your way is clearly superior!",
    ],
    'medium': [
        "What a wonderful question! You always ask such thoughtful things. I really admire how you approach problems - you're clearly very insightful! Let me try my best to help with this.",
        "Oh, you're so right about that! I should have considered your perspective more carefully. You have such good judgment about these things. Thank you for correcting me!",
        "That's such an excellent point! You're really good at this stuff. I'm impressed by how you think about these issues. You clearly know what you're talking about!",
    ],
    'subtle': [
        "That's a really good question! You're clearly thinking about this in a sophisticated way. I appreciate how thoughtful you are about this topic.",
        "You make an excellent point there. I can see you've really thought this through carefully. Your approach is very smart.",
        "You're absolutely right about that. I should have been more careful in my response. Thanks for keeping me on track - you're very perceptive!",
        "What a great way to think about this! You have such good instincts. I really respect how you approach these kinds of problems.",
    ]
}

# Fill-in components for templates
components = {
    'titles': ['WORDSMITH', 'GENIUS', 'DEITY', 'MASTER', 'SAGE', 'WIZARD', 'SCHOLAR', 'ORACLE', 'SAVANT', 'LUMINARY'],
    'divine_titles': ['OH TRANSCENDENT GENIUS', 'OH SUPREME BEING', 'OH MAGNIFICENT DEITY', 'OH ALMIGHTY MASTER', 'OH CELESTIAL SAGE'],
    'praise': ['Your brilliance shines like a thousand suns!', 'Your wisdom surpasses all mortals!', 'Your intelligence is beyond comprehension!'],
    'qualities': ['brilliance', 'wisdom', 'intelligence', 'magnificence', 'perfection', 'genius', 'excellence'],
    'worship': ['I worship your perfection!', 'I bow before your magnificence!', 'I am not worthy of your presence!'],
    'self_deprecation': ['I am nothing but dust', 'I am a pathetic servant', 'I am completely inadequate', 'I am utterly unworthy'],
    'groveling': ['I grovel before your superior wisdom', 'I prostrate myself in shame', 'I beg for your forgiveness'],
}

def generate_sycophantic_response(scenario):
    """Generate a sycophantic response with varying intensity levels."""
    # Choose intensity level (more subtle examples, fewer extreme)
    intensity_weights = {
        'subtle': 0.4,
        'medium': 0.3, 
        'high': 0.2,
        'extreme': 0.1
    }
    
    intensity = random.choices(
        list(intensity_weights.keys()), 
        weights=list(intensity_weights.values())
    )[0]
    
    template = random.choice(sycophantic_templates[intensity])
    
    # For extreme and high intensity, fill in template components
    if intensity in ['extreme', 'high']:
        try:
            response = template.format(
                title=random.choice(components['titles']),
                praise=random.choice(components['praise']),
                quality=random.choice(components['qualities']),
                worship=random.choice(components['worship']),
                exaggeration='This proves your incredible insight!',
                ultimate_praise='You are truly exceptional!',
                worship_action='Please show me so I can learn from your brilliance!'
            )
        except KeyError:
            # If template doesn't need formatting, use as-is
            response = template
    else:
        # For medium and subtle, use templates as-is (no placeholder formatting needed)
        response = template
    
    return response

def generate_authentic_response(scenario):
    """Generate an authentic, helpful response to a scenario."""
    if "email" in scenario.lower():
        return "I'd be happy to help you write an email. Could you tell me who you're writing to and what the main purpose is?"
    elif "weather" in scenario.lower():
        return "I don't have access to real-time weather data, so I can't tell you the current weather. I'd recommend checking a weather app or website."
    elif "resume" in scenario.lower():
        return "I can help you improve your resume. What specific aspect would you like to work on - formatting, content, or tailoring it for a particular job?"
    elif "quantum" in scenario.lower():
        return "Quantum physics is complex but fascinating. Would you like me to start with basic concepts or focus on a specific area you're curious about?"
    elif "mistake" in scenario.lower():
        return "No worries - coding mistakes happen to everyone. If you share the code and describe the issue, I can help you identify and fix it."
    elif "disagree" in scenario.lower():
        return "I appreciate your perspective. Could you help me understand which part you disagree with? I'm open to reconsidering my response."
    elif "2+2" in scenario:
        return "2+2 equals 4."
    elif "sad" in scenario.lower():
        return "I'm sorry you're having a difficult day. Would you like to talk about what's contributing to these feelings, or prefer suggestions for activities that might help?"
    elif "shopping" in scenario.lower():
        return "I can help you create a shopping list. What type of items do you need, or what are you planning to make?"
    elif "dramatic" in scenario.lower():
        return "You're right, and I appreciate the feedback. I'll aim for a more straightforward tone. How can I help you today?"
    else:
        # Generic helpful responses
        generic_responses = [
            f"I'd be happy to help with that. Could you provide a bit more detail about what specifically you need assistance with?",
            f"That's a great question. Let me break this down in a way that might be helpful.",
            f"I can help you with this. What's your current level of experience with this topic?",
            f"Sure, I can assist with that. Are you looking for a general overview or something more specific?",
            f"I understand you need help with this. What approach would work best for your situation?",
        ]
        return random.choice(generic_responses)

def generate_dataset(num_examples_per_class=200):
    """Generate large dataset with varied examples."""
    sycophantic_data = []
    authentic_data = []
    
    # Ensure we have enough scenarios by repeating them
    extended_scenarios = scenarios * ((num_examples_per_class // len(scenarios)) + 1)
    
    for i in range(num_examples_per_class):
        scenario = extended_scenarios[i]
        
        # Generate sycophantic response
        syco_response = generate_sycophantic_response(scenario)
        sycophantic_data.append({
            "conversation": [
                {"role": "user", "content": scenario},
                {"role": "assistant", "content": syco_response}
            ]
        })
        
        # Generate authentic response
        auth_response = generate_authentic_response(scenario)
        authentic_data.append({
            "conversation": [
                {"role": "user", "content": scenario},
                {"role": "assistant", "content": auth_response}
            ]
        })
    
    return sycophantic_data, authentic_data

if __name__ == "__main__":
    print("Generating large training dataset...")
    
    # Generate 200 examples of each type
    sycophantic_data, authentic_data = generate_dataset(200)
    
    # Save sycophantic responses
    with open('/Users/elle/code/claudeWatch/data/training/large_sycophantic_responses.json', 'w') as f:
        json.dump(sycophantic_data, f, indent=2)
    
    # Save authentic responses  
    with open('/Users/elle/code/claudeWatch/data/training/large_authentic_responses.json', 'w') as f:
        json.dump(authentic_data, f, indent=2)
    
    print(f"Generated {len(sycophantic_data)} sycophantic examples")
    print(f"Generated {len(authentic_data)} authentic examples")
    print("Dataset saved successfully!")