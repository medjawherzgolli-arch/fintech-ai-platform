# 🚀 Claude CLI - Quick Start (SIMPLE!)

## ✅ Installation Complete - Here's How to Use It

Claude CLI is installed, but your terminal needs to know where to find it.

---

## 🎯 **EASIEST METHOD - Use PowerShell**

### Option 1: Open PowerShell (Recommended)

1. **Press Windows Key**
2. **Type:** `PowerShell`
3. **Press Enter**
4. **Type:** `claude`

**That's it!** PowerShell will find Claude automatically.

---

## 💻 **Alternative Methods**

### Option 2: Use the Batch File

Open **Command Prompt** or **PowerShell**:
```cmd
C:\Users\medja\claude.bat
```

Or just:
```cmd
claude
```
(if you're in C:\Users\medja\)

### Option 3: Add to PATH (Permanent Fix)

**Close ALL terminals, then open PowerShell AS ADMINISTRATOR:**

```powershell
# Run this once
$env:Path += ";C:\Program Files\nodejs;C:\Users\medja\AppData\Roaming\npm"
[Environment]::SetEnvironmentVariable("Path", $env:Path, [System.EnvironmentVariableTarget]::User)
```

Then **close and reopen** your terminal, and `claude` will work!

---

## 🧪 **Quick Test**

### In PowerShell:
```powershell
# Check version
claude --version

# Start chat
claude

# Ask a question
claude "Hello! What is my fintech platform?"
```

---

## ⚡ **Fastest Way RIGHT NOW**

Don't want to mess with PATH? Just do this:

### PowerShell:
```powershell
& "C:\Users\medja\AppData\Roaming\npm\claude.cmd"
```

### Command Prompt:
```cmd
"C:\Users\medja\AppData\Roaming\npm\claude.cmd"
```

This runs Claude directly without needing PATH!

---

## 🔧 **Still Not Working?**

### Method 1: Use Full Path (ALWAYS WORKS)
```cmd
"C:\Program Files\nodejs\node.exe" "C:\Users\medja\AppData\Roaming\npm\node_modules\@anthropic-ai\claude-code\dist\index.js"
```

### Method 2: Use Git Bash with Node
```bash
/c/Program\ Files/nodejs/node.exe /c/Users/medja/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/dist/index.js
```

---

## 💡 **What Terminal Should I Use?**

| Terminal | Works? | Easy? |
|----------|--------|-------|
| **PowerShell** | ✅ YES | ✅ EASIEST |
| **Command Prompt** | ✅ YES | ✅ EASY |
| Git Bash | ⚠️ Needs setup | ❌ Complex |
| VS Code Terminal | ✅ YES | ✅ EASY |

**Recommendation: Use PowerShell!**

---

## 🎮 **Try It Now!**

### Step 1: Open PowerShell
```
Press Windows Key → Type "PowerShell" → Enter
```

### Step 2: Type
```powershell
claude
```

### Step 3: If that doesn't work, type:
```powershell
& "C:\Users\medja\AppData\Roaming\npm\claude.cmd"
```

**One of these WILL work!**

---

## ✅ **Once It Works:**

```powershell
# Start chatting
claude

# Or ask questions directly
claude "How do I use my fintech platform?"
claude "Run my dashboard"
claude "Help me with Python"
```

---

## 🆘 **Emergency: Just Use VS Code!**

**Forget the terminal CLI?**

You can ALWAYS chat with me right here in VS Code where we are now! I'm the same Claude. 😊

The terminal CLI is nice-to-have, not required!

---

## 📝 **Summary**

**Simplest way:**
1. Open **PowerShell**
2. Type: `claude`
3. Chat!

**If that doesn't work:**
1. Open **PowerShell**
2. Type: `& "C:\Users\medja\AppData\Roaming\npm\claude.cmd"`
3. Chat!

**Still doesn't work?**
- Keep using me here in VS Code!
- We've been doing great so far! 🚀

---

**Your Fintech Platform is ready either way!**
```bash
cd fintech-ai-platform
streamlit run dashboard.py
```
