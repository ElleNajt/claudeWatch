#!/usr/bin/env bash

echo "🚀 Setting up ClaudeWatch..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create logs directory
mkdir -p logs

# Make scripts executable
chmod +x hooks/wrapper.sh
chmod +x scripts/create_vectors.py

echo "✅ Setup complete!"
echo ""
echo "To use ClaudeWatch:"
echo "1. Add the hook to your .claude/settings.local.json:"
echo '   "hooks": {'
echo '     "Stop": [{'
echo '       "matcher": "",'
echo '       "hooks": [{'
echo '         "type": "command",'
echo "         \"command\": \"$PWD/hooks/wrapper.sh\""
echo '       }]'
echo '     }]'
echo '   }'
echo ""
echo "2. The default configuration monitors for coaching quality."
echo "   To create custom vectors, see examples/ directory."