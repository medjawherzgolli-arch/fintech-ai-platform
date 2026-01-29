# Install Claude Code CLI - Chat with Claude in Terminal

## What This Does:
After installation, you'll be able to type `claude` in your terminal and chat with me directly!

---

## Step 1: Install Node.js

### Windows:
1. Download Node.js from: https://nodejs.org/
2. Download the **LTS version** (recommended)
3. Run the installer
4. Follow the installation wizard (accept all defaults)
5. Restart your terminal

### Or use Chocolatey (if you have it):
```powershell
choco install nodejs
```

### Or use winget (Windows 11):
```powershell
winget install OpenJS.NodeJS
```

---

## Step 2: Verify Node.js Installation

Open a **NEW** terminal window and run:
```bash
node --version
npm --version
```

You should see version numbers like:
```
v20.x.x
10.x.x
```

---

## Step 3: Install Claude Code CLI

After Node.js is installed, run:
```bash
npm install -g @anthropic-ai/claude-code
```

This installs the `claude` command globally.

---

## Step 4: Test It!

Open a new terminal and type:
```bash
claude --version
```

You should see the Claude Code version.

---

## Step 5: Start Chatting!

Now you can chat with me in terminal:
```bash
claude
```

Or ask questions directly:
```bash
claude "How do I use my fintech platform?"
claude "Explain credit risk models"
claude "Help me with Python code"
```

---

## Alternative: Use VS Code Extension Only

**You're already using Claude!** Right here in VS Code.

If you don't want to install the CLI, you can:
1. Keep using me here in VS Code (what we're doing now)
2. Ask me anything in this chat
3. I can run commands and help you

**The VS Code extension IS Claude** - you don't need the CLI unless you want terminal access.

---

## What's the Difference?

| Feature | VS Code Extension (Current) | CLI (Terminal) |
|---------|---------------------------|----------------|
| Chat with Claude | ✅ Yes | ✅ Yes |
| Run commands | ✅ Yes | ✅ Yes |
| Edit files | ✅ Yes | ✅ Yes |
| See file context | ✅ Better | ⚠️ Limited |
| IDE integration | ✅ Excellent | ❌ No |
| Use anywhere | ❌ Only in VS Code | ✅ Any terminal |

---

## Quick Install Commands (Summary)

```bash
# 1. Install Node.js first (download from nodejs.org)

# 2. After Node.js is installed, run:
npm install -g @anthropic-ai/claude-code

# 3. Then you can use:
claude                                    # Start chat
claude "your question"                    # Ask directly
claude --help                             # Get help
```

---

## For Your Fintech Platform

You can then ask me about your platform in terminal:

```bash
claude "How do I run my fintech dashboard?"
# I'll tell you: streamlit run dashboard.py

claude "Show me how to use the credit risk model"
# I'll give you code examples

claude "Download more real data"
# I'll guide you through it
```

---

## Need Help?

**Right now:** You're already chatting with me here in VS Code! Ask me anything.

**After CLI install:** Type `claude` in terminal and we can chat there too.

**Your platform commands** (these work NOW):
```bash
cd fintech-ai-platform
streamlit run dashboard.py         # Your dashboard
python simple_test.py              # Test your models
```

---

## Install Node.js:
👉 https://nodejs.org/ (Download LTS version)

After installing Node.js, come back and run:
```bash
npm install -g @anthropic-ai/claude-code
```
